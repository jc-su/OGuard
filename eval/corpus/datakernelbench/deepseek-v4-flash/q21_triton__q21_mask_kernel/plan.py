import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
import numpy as np

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _q21_mask_kernel(
    l1_orderkey_ptr, l1_late_ptr, supp_ids_ptr,
    lo_orderkeys_ptr, lo_distinct_supp_ptr, lo_late_distinct_supp_ptr,
    precomputed_idx_ptr,
    mask_out_ptr, n_elements, num_suppliers,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    l1_orderkey = tl.load(l1_orderkey_ptr + offs, mask=mask, other=-1)
    l1_late = tl.load(l1_late_ptr + offs, mask=mask, other=0)
    supp_id = tl.load(supp_ids_ptr + offs, mask=mask, other=-1)
    idx = tl.load(precomputed_idx_ptr + offs, mask=mask, other=0)
    
    distinct_supp = tl.load(lo_distinct_supp_ptr + idx, mask=mask, other=0)
    late_distinct_supp = tl.load(lo_late_distinct_supp_ptr + idx, mask=mask, other=0)
    
    exists_other = distinct_supp >= 2
    not_exists_other_late = late_distinct_supp <= 1
    
    keep = mask & (l1_late != 0) & exists_other & not_exists_other_late & (supp_id >= 0) & (supp_id < num_suppliers)
    tl.store(mask_out_ptr + offs, tl.where(keep, 1, 0), mask=mask)
'''

def _load_kernels():
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

def run_query(tables: dict[str, cudf.DataFrame], nation_name: str = "SAUDI ARABIA", orderstatus: str = "F") -> cudf.DataFrame:
    """
    SQL: TPC-H Q21 - Suppliers Who Kept Orders Waiting.
    Description: cuDF for joins/strings/filter, then GPU (fused Triton) for the core EXISTS/NOT EXISTS
                 filter and aggregation. Faster by fusing the row-wise conditions into one kernel launch,
                 avoiding multiple passes over lineitem data.
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
    l1_late_t = torch.as_tensor(l1["late"].to_cupy(), dtype=torch.bool)
    supp_ids_t = torch.as_tensor(l1["supp_id"].to_cupy(), dtype=torch.int64)

    lo_orderkeys_t = torch.as_tensor(agg["l_orderkey"].to_cupy(), dtype=torch.int64)
    lo_distinct_supp_t = torch.as_tensor(agg["distinct_supp"].to_cupy(), dtype=torch.int64)
    lo_late_distinct_supp_t = torch.as_tensor(agg["late_distinct_supp"].to_cupy(), dtype=torch.int64)

    num_suppliers = len(supp_n)
    n = l1_orderkey_t.numel()

    device = l1_orderkey_t.device

    idx_t = torch.searchsorted(lo_orderkeys_t, l1_orderkey_t)

    mask_out = torch.zeros(n, dtype=torch.int32, device=device)
    # Use a default grid with BLOCK_SIZE=4096
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _K._q21_mask_kernel[grid](
        l1_orderkey_t, l1_late_t, supp_ids_t,
        lo_orderkeys_t, lo_distinct_supp_t, lo_late_distinct_supp_t,
        idx_t, mask_out, n, num_suppliers,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    mask_bool = mask_out.bool()
    valid_supp_ids = supp_ids_t[mask_bool]

    counts_t = torch.zeros(num_suppliers, dtype=torch.int64, device=device)
    if valid_supp_ids.numel() > 0:
        ones = torch.ones_like(valid_supp_ids, dtype=torch.int64)
        counts_t.scatter_add_(0, valid_supp_ids.to(torch.int64), ones)

    supp_idx_t = torch.arange(num_suppliers, device=device)

    valid_mask = counts_t > 0
    valid_supp_idx = supp_idx_t[valid_mask]
    valid_counts = counts_t[valid_mask]

    valid_supp_cpu = valid_supp_idx.cpu().numpy()
    valid_counts_cpu = valid_counts.cpu().numpy()
    
    result = cudf.DataFrame({
        "s_name": supp_n["s_name"].iloc[valid_supp_cpu].reset_index(drop=True),
        "numwait": cudf.Series(valid_counts_cpu),
    })
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100).reset_index(drop=True)
    return result
