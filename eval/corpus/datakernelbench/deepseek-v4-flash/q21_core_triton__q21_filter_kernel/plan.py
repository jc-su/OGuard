import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
import numpy as np

_KERNEL_SOURCE = r"""
import triton
import triton.language as tl

@triton.jit
def _q21_filter_kernel(
    l1_orderkey_ptr,
    l1_late_ptr,
    supp_ids_ptr,
    lo_orderkeys_ptr,
    lo_distinct_supp_ptr,
    lo_late_distinct_supp_ptr,
    out_valid_ptr,     # int64: store valid supp_id (or -1 if not valid)
    out_count_ptr,     # int64: store 1 for valid, 0 for invalid
    n_l1,
    n_lo,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_l1

    l1_ok = tl.load(l1_orderkey_ptr + offs, mask=mask, other=0)
    l1_late_raw = tl.load(l1_late_ptr + offs, mask=mask, other=0)
    l1_late = l1_late_raw.to(tl.int1)
    supp_id = tl.load(supp_ids_ptr + offs, mask=mask, other=-1)

    lo = tl.full((BLOCK_SIZE,), 0, dtype=tl.int64)
    hi = tl.full((BLOCK_SIZE,), n_lo, dtype=tl.int64)
    for _ in tl.static_range(24):
        mid = (lo + hi) // 2
        mid_val = tl.load(lo_orderkeys_ptr + mid, mask=mask, other=0)
        mid_val_lt = mid_val < l1_ok
        lo = tl.where(mid_val_lt & mask, mid + 1, lo)
        hi = tl.where(~mid_val_lt & mask, mid, hi)
    idx = lo
    idx = tl.minimum(idx, n_lo - 1)

    distinct_supp = tl.load(lo_distinct_supp_ptr + idx, mask=mask, other=0)
    late_distinct_supp = tl.load(lo_late_distinct_supp_ptr + idx, mask=mask, other=0)

    exists_other = distinct_supp >= 2
    not_exists_other_late = late_distinct_supp <= 1

    keep = mask & l1_late & exists_other & not_exists_other_late & (supp_id >= 0)

    tl.store(out_valid_ptr + offs, tl.where(keep, supp_id, -1), mask=mask)
    tl.store(out_count_ptr + offs, tl.where(keep, 1, 0), mask=mask)
"""

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q21_triton_")
    path = os.path.join(tmpdir, "_q21_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q21_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

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
    n_l1 = l1_orderkey.numel()
    device = l1_orderkey.device
    n_lo = lo_orderkeys.numel()

    if n_l1 == 0:
        return torch.arange(num_suppliers, device=device), torch.zeros(num_suppliers, dtype=torch.int64, device=device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_l1, BLOCK_SIZE),)

    valid_supp = torch.full((n_l1,), -1, dtype=torch.int64, device=device)
    valid_count = torch.zeros(n_l1, dtype=torch.int64, device=device)

    _K._q21_filter_kernel[grid](
        l1_orderkey.contiguous(),
        l1_late.to(torch.int8).contiguous(),
        supp_ids.contiguous(),
        lo_orderkeys.contiguous(),
        lo_distinct_supp.contiguous(),
        lo_late_distinct_supp.contiguous(),
        valid_supp,
        valid_count,
        n_l1,
        n_lo,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Use torch scatter_add for histogram
    mask_valid = valid_supp >= 0
    counts = torch.zeros(num_suppliers, dtype=torch.int64, device=device)
    counts.scatter_add_(0, valid_supp[mask_valid].to(torch.int64), valid_count[mask_valid])

    return torch.arange(num_suppliers, device=device), counts

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

    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supplier.merge(nation_f, left_on="s_nationkey", right_on="n_nationkey")
    supp_n = supp_n[["s_suppkey", "s_name"]].sort_values("s_name").reset_index(drop=True)
    supp_n["supp_id"] = cudf.Series(range(len(supp_n)), dtype="int32")

    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]

    li_all = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        orders_f, left_on="l_orderkey", right_on="o_orderkey"
    )
    li_all["late"] = (li_all["l_receiptdate"] > li_all["l_commitdate"]).astype("int8")

    distinct_supp = (
        li_all[["l_orderkey", "l_suppkey"]]
        .drop_duplicates()
        .groupby("l_orderkey")
        .size()
        .reset_index()
    )
    distinct_supp.columns = ["l_orderkey", "distinct_supp"]

    late_rows = li_all[li_all["late"] == 1][["l_orderkey", "l_suppkey"]].drop_duplicates()
    late_distinct = late_rows.groupby("l_orderkey").size().reset_index()
    late_distinct.columns = ["l_orderkey", "late_distinct_supp"]

    agg = distinct_supp.merge(late_distinct, on="l_orderkey", how="left")
    agg["late_distinct_supp"] = agg["late_distinct_supp"].fillna(0).astype("int32")
    agg["distinct_supp"] = agg["distinct_supp"].astype("int32")
    agg = agg.sort_values("l_orderkey").reset_index(drop=True)

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
