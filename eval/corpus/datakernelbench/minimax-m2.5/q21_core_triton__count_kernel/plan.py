"""TPC‑H Q21 – optimized with Triton (Level: core)."""

import os
import tempfile
import importlib.util
import linecache

import torch
import triton
import cudf

# ----------------------------------------------------------------------
# Triton kernel source – must reside in a separate on‑disk file for
# `importlib.util.spec_from_file_location` to work.
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _count_kernel(supp_ids_ptr, mask_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused predicate evaluation + per‑supplier aggregation."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    # Load supplier id for this row (fallback -1 for out‑of‑bounds)
    sid = tl.load(supp_ids_ptr + offs, mask=valid, other=-1)

    # Load the pre‑computed boolean mask (true if row satisfies Q21 predicates)
    m = tl.load(mask_ptr + offs, mask=valid, other=0)

    # Combine with validity of supplier id
    cond = m & (sid >= 0)

    # Safe index – we add zero when condition is false to avoid invalid addresses
    safe_sid = tl.where(sid >= 0, sid, 0)
    add = tl.where(cond, 1, 0)

    tl.atomic_add(out_ptr + safe_sid, add)
'''


def _load_kernels():
    """Write the kernel source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="q21_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


# ----------------------------------------------------------------------
# Core compute – replaces the original PyTorch scatter_add with a Triton
# kernel that evaluates the filter predicate and performs atomic addition
# in a single fused pass.
# ----------------------------------------------------------------------
def _query_core(
    l1_orderkey: torch.Tensor,
    l1_suppkey: torch.Tensor,  # unused, kept for signature compatibility
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    lo_orderkeys: torch.Tensor,
    lo_distinct_supp: torch.Tensor,
    lo_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per‑supplier counts for Q21.

    This version:
      1. Uses ``torch.searchsorted`` to map each l1 row to its order‑level
         aggregated statistics (distinct suppliers, late distinct suppliers).
      2. Builds the combined predicate on the GPU.
      3. Launches a Triton kernel that evaluates the predicate and performs
         atomic addition into the per‑supplier counter in one pass.

    The kernel is memory‑bound; fusion reduces HBM traffic by avoiding a
    separate scatter‑add kernel and by keeping the predicate evaluation
    close to the aggregation.
    """
    # Map l1 rows to the aggregated statistics of their orderkey.
    idx = torch.searchsorted(lo_orderkeys, l1_orderkey)
    distinct_row = lo_distinct_supp[idx]          # int64
    late_distinct_row = lo_late_distinct_supp[idx]  # int64

    # Q21 predicate: l1 is late, another supplier exists,
    # no other supplier is late, and the supplier id is valid.
    mask = (
        l1_late
        & (distinct_row >= 2)
        & (late_distinct_row <= 1)
        & (supp_ids >= 0)
    )

    # Allocate per‑supplier counter and zero‑initialize (handled by autotune).
    counts = torch.zeros(num_suppliers, dtype=torch.int64, device=supp_ids.device)

    # Launch the fused kernel.
    n_rows = supp_ids.shape[0]
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _K._count_kernel[grid](supp_ids, mask, counts, n_rows)

    # Return supplier indices and the computed counts.
    supp_idx = torch.arange(num_suppliers, device=counts.device, dtype=torch.int64)
    return supp_idx, counts


# ----------------------------------------------------------------------
# Query runner – unchanged from the reference implementation.
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "SAUDI ARABIA",
    orderstatus: str = "F",
) -> cudf.DataFrame:
    """
    SQL: TPC-H Q21 - Suppliers Who Kept Orders Waiting.
    Description: For each supplier in the given nation, count orders of status 'F' where the
                 supplier's lineitem was late, another supplier also shipped the order, and no
                 other supplier was late on that order.
    Params:
      nation_name (str): nation name filter.
      orderstatus (str): order status filter.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # Filter nation -> supplier
    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supplier.merge(nation_f, left_on="s_nationkey", right_on="n_nationkey")
    supp_n = supp_n[["s_suppkey", "s_name"]].sort_values("s_name").reset_index(drop=True)
    supp_n["supp_id"] = cudf.Series(range(len(supp_n)), dtype="int32")

    # Filter orders by status
    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]

    # --- Compute per-orderkey aggregates over FULL lineitem (restricted to orders_f) ---
    li_all = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        orders_f, left_on="l_orderkey", right_on="o_orderkey"
    )
    li_all["late"] = (li_all["l_receiptdate"] > li_all["l_commitdate"]).astype("int8")

    # distinct suppliers per orderkey
    distinct_supp = (
        li_all[["l_orderkey", "l_suppkey"]]
        .drop_duplicates()
        .groupby("l_orderkey")
        .size()
        .reset_index()
    )
    distinct_supp.columns = ["l_orderkey", "distinct_supp"]

    # distinct late suppliers per orderkey
    late_rows = li_all[li_all["late"] == 1][["l_orderkey", "l_suppkey"]].drop_duplicates()
    late_distinct = late_rows.groupby("l_orderkey").size().reset_index()
    late_distinct.columns = ["l_orderkey", "late_distinct_supp"]

    agg = distinct_supp.merge(late_distinct, on="l_orderkey", how="left")
    agg["late_distinct_supp"] = agg["late_distinct_supp"].fillna(0).astype("int32")
    agg["distinct_supp"] = agg["distinct_supp"].astype("int32")
    agg = agg.sort_values("l_orderkey").reset_index(drop=True)

    # --- Build l1 rows: supplier join + orders filter ---
    l1 = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        supp_n, left_on="l_suppkey", right_on="s_suppkey"
    )
    l1 = l1.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")
    l1["late"] = (l1["l_receiptdate"] > l1["l_commitdate"]).astype("bool")

    l1_orderkey_t = torch.as_tensor(l1["l_orderkey"].to_cupy(), dtype=torch.int64)
    l1_suppkey_t = torch.as_tensor(l1["l_suppkey"].to_cupy(), dtype=torch.int64)
    l1_late_t = torch.as_tensor(l1["late"].to_cupy(), dtype=torch.bool)
    supp_ids_t = torch.as_tensor(l1["supp_id"].to_cupy(), dtype=torch.int64)

    lo_orderkeys_t = torch.as_tensor(agg["l_orderkey"].to_cupy(), dtype=torch.int64)
    lo_distinct_supp_t = torch.as_tensor(agg["distinct_supp"].to_cupy(), dtype=torch.int64)
    lo_late_distinct_supp_t = torch.as_tensor(agg["late_distinct_supp"].to_cupy(), dtype=torch.int64)

    num_suppliers = len(supp_n)

    supp_idx_t, counts_t = _query_core(
        l1_orderkey_t,
        l1_suppkey_t,
        l1_late_t,
        supp_ids_t,
        lo_orderkeys_t,
        lo_distinct_supp_t,
        lo_late_distinct_supp_t,
        num_suppliers,
    )

    result = cudf.DataFrame({
        "s_name": supp_n["s_name"].iloc[cudf.Series(supp_idx_t)].reset_index(drop=True),
        "numwait": cudf.Series(counts_t),
    })
    result = result[result["numwait"] > 0]
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100).reset_index(drop=True)
    return result
