import os
import tempfile
import torch
import cudf
import pandas as pd
import numpy as np
from torch.utils.cpp_extension import load
import linecache

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void q21_full_kernel(
    const int64_t* __restrict__ l1_orderkey,
    const int64_t* __restrict__ l1_late_int,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int64_t* __restrict__ lo_distinct_supp,
    const int64_t* __restrict__ lo_late_distinct_supp,
    int64_t* __restrict__ counts,
    int64_t n_elements,
    int64_t n_lo
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;
    
    int64_t ok = l1_orderkey[tid];
    if (l1_late_int[tid] == 0) return;
    int64_t sid = supp_ids[tid];
    if (sid < 0) return;
    
    // Optimized binary search with early exit
    int64_t left = 0, right = n_lo - 1;
    while (left <= right) {
        int64_t mid = left + (right - left) / 2;
        int64_t mid_val = lo_orderkeys[mid];
        if (mid_val == ok) {
            int64_t distinct = lo_distinct_supp[mid];
            int64_t late_distinct = lo_late_distinct_supp[mid];
            if (distinct >= 2 && late_distinct <= 1) {
                atomicAdd((unsigned long long*)&counts[sid], 1ULL);
            }
            return;
        } else if (mid_val < ok) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
}

torch::Tensor q21_launcher(
    torch::Tensor l1_orderkey,
    torch::Tensor l1_late_int,
    torch::Tensor supp_ids,
    torch::Tensor lo_orderkeys,
    torch::Tensor lo_distinct_supp,
    torch::Tensor lo_late_distinct_supp,
    int64_t num_suppliers
) {
    if (!l1_orderkey.is_cuda()) l1_orderkey = l1_orderkey.cuda();
    if (!l1_late_int.is_cuda()) l1_late_int = l1_late_int.cuda();
    if (!supp_ids.is_cuda()) supp_ids = supp_ids.cuda();
    if (!lo_orderkeys.is_cuda()) lo_orderkeys = lo_orderkeys.cuda();
    if (!lo_distinct_supp.is_cuda()) lo_distinct_supp = lo_distinct_supp.cuda();
    if (!lo_late_distinct_supp.is_cuda()) lo_late_distinct_supp = lo_late_distinct_supp.cuda();
    
    l1_orderkey = l1_orderkey.contiguous();
    l1_late_int = l1_late_int.contiguous();
    supp_ids = supp_ids.contiguous();
    lo_orderkeys = lo_orderkeys.contiguous();
    lo_distinct_supp = lo_distinct_supp.contiguous();
    lo_late_distinct_supp = lo_late_distinct_supp.contiguous();
    
    int64_t n_elements = l1_orderkey.numel();
    int64_t n_lo = lo_orderkeys.numel();
    
    auto counts = torch::zeros({num_suppliers}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    int blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 65535) blocks = 65535;
    
    q21_full_kernel<<<blocks, BLOCK_SIZE>>>(
        l1_orderkey.data_ptr<int64_t>(),
        l1_late_int.data_ptr<int64_t>(),
        supp_ids.data_ptr<int64_t>(),
        lo_orderkeys.data_ptr<int64_t>(),
        lo_distinct_supp.data_ptr<int64_t>(),
        lo_late_distinct_supp.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(),
        n_elements,
        n_lo
    );
    
    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q21_launcher", &q21_launcher, "TPC-H Q21 (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q21_cuda_")
    cu_path = os.path.join(tmpdir, "_q21_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    linecache.checkcache(cu_path)
    return load(
        name="q21_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

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
    l1_late_int = l1_late.to(torch.int64)
    counts = _EXT.q21_launcher(
        l1_orderkey, l1_late_int, supp_ids,
        lo_orderkeys, lo_distinct_supp, lo_late_distinct_supp,
        num_suppliers
    )
    return torch.arange(num_suppliers, device=counts.device), counts


def run_query(tables: dict[str, cudf.DataFrame], nation_name: str = "SAUDI ARABIA", orderstatus: str = "F") -> cudf.DataFrame:
    """
    SQL: TPC-H Q21 - Suppliers Who Kept Orders Waiting.
    Uses cuDF for joins/aggregations and a single fused CUDA kernel for filtering+counting.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # Optimized: filter nation, supplier, orders first, then merge only what's needed
    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supplier.merge(nation_f, left_on="s_nationkey", right_on="n_nationkey")[["s_suppkey", "s_name"]]
    supp_n = supp_n.sort_values("s_name").reset_index(drop=True)
    supp_n["supp_id"] = cudf.Series(range(len(supp_n)), dtype="int32")
    
    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]

    # Optimized: single pass for order-level aggregates
    li_temp = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]]
    li_temp = li_temp.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")
    
    # Use vectorized operations instead of astype
    li_late = (li_temp["l_receiptdate"] > li_temp["l_commitdate"])
    
    # Efficient groupby aggregates
    li_grp = li_temp.groupby("l_orderkey")
    distinct_supp = li_grp["l_suppkey"].nunique().reset_index()
    distinct_supp.columns = ["l_orderkey", "distinct_supp"]
    
    # Efficient distinct late suppliers
    late_mask = li_late
    li_late_grp = li_temp[late_mask].groupby("l_orderkey")
    late_distinct = li_late_grp["l_suppkey"].nunique().reset_index()
    late_distinct.columns = ["l_orderkey", "late_distinct_supp"]
    
    agg = distinct_supp.merge(late_distinct, on="l_orderkey", how="left").fillna(0)
    agg = agg.sort_values("l_orderkey").reset_index(drop=True)
    agg["distinct_supp"] = agg["distinct_supp"].astype("int64")
    agg["late_distinct_supp"] = agg["late_distinct_supp"].astype("int64")

    # Build l1: supplier line items
    l1 = li_temp.merge(supp_n, left_on="l_suppkey", right_on="s_suppkey")
    
    # Convert to tensors efficiently
    l1_orderkey_t = torch.as_tensor(l1["l_orderkey"].to_cupy(), dtype=torch.int64)
    l1_suppkey_t = torch.as_tensor(l1["l_suppkey"].to_cupy(), dtype=torch.int64)
    l1_late_t = torch.as_tensor((l1["l_receiptdate"] > l1["l_commitdate"]).to_cupy(), dtype=torch.bool)
    supp_ids_t = torch.as_tensor(l1["supp_id"].to_cupy(), dtype=torch.int64)
    
    lo_orderkeys_t = torch.as_tensor(agg["l_orderkey"].to_cupy(), dtype=torch.int64)
    lo_distinct_supp_t = torch.as_tensor(agg["distinct_supp"].to_cupy(), dtype=torch.int64)
    lo_late_distinct_supp_t = torch.as_tensor(agg["late_distinct_supp"].to_cupy(), dtype=torch.int64)
    
    num_suppliers = len(supp_n)
    
    supp_idx_t, counts_t = _query_core(
        l1_orderkey_t, l1_suppkey_t, l1_late_t, supp_ids_t,
        lo_orderkeys_t, lo_distinct_supp_t, lo_late_distinct_supp_t,
        num_suppliers,
    )
    
    result = cudf.DataFrame({
        "s_name": supp_n["s_name"].iloc[cudf.Series(supp_idx_t.cpu())].reset_index(drop=True),
        "numwait": cudf.Series(counts_t.cpu()),
    })
    result = result[result["numwait"] > 0]
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100).reset_index(drop=True)
    return result
