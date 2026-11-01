"""Triton core replacement for TPC-H Q21.

The public cuDF preparation and result assembly remain in ``run_query`` exactly as in
the reference plan.  The tensor hot path in ``_query_core`` runs on the GPU in one
fused Triton pass: per candidate l1 row it performs the sorted order-key lookup,
applies the EXISTS / NOT EXISTS predicates, and atomically accumulates supplier
counts.  This avoids separate eager Torch searchsorted, masking, temporary tensor,
and scatter-add passes for this memory-sensitive scan.
"""

import os
import tempfile
import importlib.util
import linecache
from types import ModuleType

import torch
import triton
import cudf


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q21_core_kernel(
    l1_orderkey_ptr,
    l1_late_ptr,
    supp_ids_ptr,
    lo_orderkeys_ptr,
    lo_distinct_supp_ptr,
    lo_late_distinct_supp_ptr,
    counts_ptr,
    n_l1,
    lo_n,
    BLOCK_SIZE: tl.constexpr,
    SEARCH_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_l1

    late = tl.load(l1_late_ptr + offs, mask=row_mask, other=0)
    supp_id = tl.load(supp_ids_ptr + offs, mask=row_mask, other=-1)

    # Filter early so non-late rows do not issue random binary-search loads.
    cand = row_mask & (late != 0) & (supp_id >= 0)
    key = tl.load(l1_orderkey_ptr + offs, mask=cand, other=0)

    low = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    high = low + lo_n

    # Vectorized lower_bound(searchsorted) over sorted unique lo_orderkeys.
    for _ in tl.static_range(0, SEARCH_ITERS):
        mid = (low + high) // 2
        mid_valid = cand & (mid < lo_n)
        mid_key = tl.load(
            lo_orderkeys_ptr + mid,
            mask=mid_valid,
            other=9223372036854775807,
        )
        go_right = mid_key < key
        low = tl.where(cand & go_right, mid + 1, low)
        high = tl.where(cand & (~go_right), mid, high)

    idx = low
    stat_mask = cand & (idx < lo_n)
    distinct_supp = tl.load(lo_distinct_supp_ptr + idx, mask=stat_mask, other=0)
    late_distinct_supp = tl.load(lo_late_distinct_supp_ptr + idx, mask=stat_mask, other=0)

    keep = stat_mask & (distinct_supp >= 2) & (late_distinct_supp <= 1)
    safe_supp_id = tl.where(keep, supp_id, 0)
    one = tl.full((BLOCK_SIZE,), 1, dtype=tl.int64)
    tl.atomic_add(counts_ptr + safe_supp_id, one, sem="relaxed", mask=keep)
'''


def _load_kernels() -> ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation.

    Triton requires JIT functions to originate from on-disk source in this validation
    environment.  The host module keeps cuDF/Torch orchestration, while the imported
    temporary module contains only GPU Triton kernels.
    """
    tmpdir = tempfile.mkdtemp(prefix="q21_triton_")
    path = os.path.join(tmpdir, "_q21_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q21_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import specification for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K: ModuleType = _load_kernels()


def _search_iterations(n: int) -> int:
    """Return a safe fixed iteration count for Triton lower_bound over ``n`` keys."""
    return max(1, int(n).bit_length())


def _launch_q21_core(
    l1_orderkey: torch.Tensor,
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    lo_orderkeys: torch.Tensor,
    lo_distinct_supp: torch.Tensor,
    lo_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused GPU kernel for Q21 supplier wait counts.

    The output index tensor is created with Torch, while the expensive row-wise
    search/filter/count path runs in Triton on the GPU.  Fusing the binary lookup,
    predicates, and aggregation reduces intermediate HBM traffic versus eager
    Torch searchsorted + boolean masking + scatter_add.
    """
    counts = torch.zeros(num_suppliers, dtype=torch.int64, device=supp_ids.device)
    supp_idx = torch.arange(num_suppliers, device=supp_ids.device)

    n_l1 = int(l1_orderkey.numel())
    lo_n = int(lo_orderkeys.numel())
    if n_l1 == 0 or num_suppliers <= 0:
        return supp_idx, counts

    grid = (triton.cdiv(n_l1, block_size),)
    _K._q21_core_kernel[grid](
        l1_orderkey,
        l1_late,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        counts,
        n_l1,
        lo_n,
        BLOCK_SIZE=block_size,
        SEARCH_ITERS=_search_iterations(lo_n),
        num_warps=4,
    )
    return supp_idx, counts


def _query_core(
    l1_orderkey: torch.Tensor,
    l1_suppkey: torch.Tensor,
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    lo_orderkeys: torch.Tensor,
    lo_distinct_supp: torch.Tensor,
    lo_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU tensor hot path for TPC-H Q21.

    Runs a fused Triton kernel on GPU tensors.  For each l1 row it conditionally
    performs the per-order search only for late supplier-nation candidates, checks
    the distinct-supplier and late-distinct-supplier predicates in registers, and
    atomically accumulates int64 counts by supplier id.  This should be faster than
    the reference eager Torch region because it avoids materializing search results,
    boolean masks, selected supplier ids, and a separate scatter-add pass.
    """
    return _launch_q21_core(
        l1_orderkey,
        l1_late,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        num_suppliers,
    )


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
