import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton  # needed for grid calculation (triton.cdiv)

# ----------------------------------------------------------------------
# Triton kernel source – written to disk and imported at module load time
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune configs tuned for H100 (large BLOCK_SIZE, 4‑8 warps)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ] + [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=8, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],  # required when using atomic_add
)
@triton.jit
def _count_kernel(
    late_ptr,                     # int8*  (0/1)
    supp_ptr,                     # int64*
    distinct_ptr,                 # int64*
    late_distinct_ptr,            # int64*
    out_ptr,                      # int64*   (per‑supplier counters)
    n_elements,                   # total number of rows
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load per‑row data; safe defaults for OOB lanes
    late = tl.load(late_ptr + offs, mask=mask, other=0).to(tl.int1)
    supp = tl.load(supp_ptr + offs, mask=mask, other=-1).to(tl.int64)
    distinct = tl.load(distinct_ptr + offs, mask=mask, other=0).to(tl.int64)
    late_distinct = tl.load(late_distinct_ptr + offs, mask=mask, other=0).to(tl.int64)

    # Predicate logic (identical to reference)
    keep = (
        mask
        & (late == 1)
        & (distinct >= 2)
        & (late_distinct <= 1)
        & (supp >= 0)
    )

    # Safe index for atomic add (when supp==-1 mask==False)
    safe_supp = tl.where(supp >= 0, supp, 0)

    # One atomic add of 1 per qualifying row
    tl.atomic_add(out_ptr + safe_supp, 1, mask=keep)
'''

def _load_kernels():
    """Write the Triton source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="q21_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()  # contains the compiled Triton kernels


def _launch_count_kernel(
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    distinct_per_row: torch.Tensor,
    late_distinct_per_row: torch.Tensor,
    num_suppliers: int,
) -> torch.Tensor:
    """
    Launch the fused Triton kernel that evaluates the predicate
    and performs a per‑supplier atomic count.
    """
    n = l1_late.numel()
    # Ensure dtype matches kernel expectations
    l1_late_i8 = l1_late.to(torch.int8)

    out = torch.zeros(num_suppliers, dtype=torch.int64, device=l1_late.device)

    # Grid size is computed from the autotuned BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._count_kernel[grid](
        l1_late_i8,
        supp_ids,
        distinct_per_row,
        late_distinct_per_row,
        out,
        n,
    )
    return out


def _query_core(
    l1_orderkey: torch.Tensor,
    l1_suppkey: torch.Tensor,          # unused in the hot path, kept for signature compatibility
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    lo_orderkeys: torch.Tensor,
    lo_distinct_supp: torch.Tensor,
    lo_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton‑fused version of the original hot path.

    Steps:
      1. Use Torch's `searchsorted` to map each lineitem row to its orderkey aggregate.
      2. Gather the per‑order aggregates (`distinct_supp`, `late_distinct_supp`).
      3. Launch a single fused Triton kernel that evaluates the four predicates
         and atomically increments the per‑supplier counters.

    This replaces the original sequence of tensor indexing + Python‑level mask +
    `scatter_add_` with one memory‑bound kernel, reducing HBM traffic
    and kernel launch overhead.
    """
    # 1. Locate each row's orderkey in the aggregated table
    idx = torch.searchsorted(lo_orderkeys, l1_orderkey, right=False)

    # 2. Gather aggregates per row
    distinct_per_row = lo_distinct_supp[idx]
    late_distinct_per_row = lo_late_distinct_supp[idx]

    # 3. Fused predicate + atomic count
    counts = _launch_count_kernel(
        l1_late,
        supp_ids,
        distinct_per_row,
        late_distinct_per_row,
        num_suppliers,
    )

    # Return a tensor of supplier indices and the corresponding counts
    return torch.arange(num_suppliers, device=counts.device), counts


def run_query(tables: dict[str, cudf.DataFrame], nation_name: str = "SAUDI ARABIA", orderstatus: str = "F") -> cudf.DataFrame:
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

    # Convert cuDF columns to Torch tensors on the GPU
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
