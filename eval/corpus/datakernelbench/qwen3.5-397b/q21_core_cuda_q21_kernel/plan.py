import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__device__ int64_t binary_search_left(const int64_t* arr, int64_t n, int64_t val) {
    int64_t left = 0, right = n;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        if (arr[mid] < val) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__global__ void q21_kernel(
    const int64_t* l1_orderkey,
    const int64_t* l1_suppkey,
    const bool* l1_late,
    const int64_t* supp_ids,
    const int64_t* lo_orderkeys,
    const int64_t* lo_distinct_supp,
    const int64_t* lo_late_distinct_supp,
    unsigned long long* counts,
    int64_t n_rows,
    int64_t n_orders,
    int64_t num_suppliers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;
    
    int64_t orderkey = l1_orderkey[idx];
    int64_t supp_id = supp_ids[idx];
    bool late = l1_late[idx];
    
    if (supp_id < 0) return;
    if (!late) return;
    
    int64_t order_idx = binary_search_left(lo_orderkeys, n_orders, orderkey);
    if (order_idx >= n_orders || lo_orderkeys[order_idx] != orderkey) return;
    
    int64_t distinct_supp = lo_distinct_supp[order_idx];
    int64_t late_distinct_supp = lo_late_distinct_supp[order_idx];
    
    if (distinct_supp < 2) return;
    if (late_distinct_supp > 1) return;
    
    atomicAdd(&counts[supp_id], 1ULL);
}

torch::Tensor q21_launcher(
    torch::Tensor l1_orderkey,
    torch::Tensor l1_suppkey,
    torch::Tensor l1_late,
    torch::Tensor supp_ids,
    torch::Tensor lo_orderkeys,
    torch::Tensor lo_distinct_supp,
    torch::Tensor lo_late_distinct_supp,
    int64_t num_suppliers
) {
    TORCH_CHECK(l1_orderkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(l1_suppkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(l1_late.scalar_type() == torch::kBool);
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64);
    TORCH_CHECK(lo_orderkeys.scalar_type() == torch::kInt64);
    TORCH_CHECK(lo_distinct_supp.scalar_type() == torch::kInt64);
    TORCH_CHECK(lo_late_distinct_supp.scalar_type() == torch::kInt64);
    
    if (!l1_orderkey.is_cuda()) l1_orderkey = l1_orderkey.cuda();
    if (!l1_suppkey.is_cuda()) l1_suppkey = l1_suppkey.cuda();
    if (!l1_late.is_cuda()) l1_late = l1_late.cuda();
    if (!supp_ids.is_cuda()) supp_ids = supp_ids.cuda();
    if (!lo_orderkeys.is_cuda()) lo_orderkeys = lo_orderkeys.cuda();
    if (!lo_distinct_supp.is_cuda()) lo_distinct_supp = lo_distinct_supp.cuda();
    if (!lo_late_distinct_supp.is_cuda()) lo_late_distinct_supp = lo_late_distinct_supp.cuda();
    
    l1_orderkey = l1_orderkey.contiguous();
    l1_suppkey = l1_suppkey.contiguous();
    l1_late = l1_late.contiguous();
    supp_ids = supp_ids.contiguous();
    lo_orderkeys = lo_orderkeys.contiguous();
    lo_distinct_supp = lo_distinct_supp.contiguous();
    lo_late_distinct_supp = lo_late_distinct_supp.contiguous();
    
    int64_t n_rows = l1_orderkey.numel();
    int64_t n_orders = lo_orderkeys.numel();
    
    auto counts = torch::zeros({num_suppliers}, torch::dtype(torch::kInt64).device(l1_orderkey.device()));
    
    int blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q21_kernel<<<blocks, BLOCK_SIZE>>>(
        l1_orderkey.data_ptr<int64_t>(),
        l1_suppkey.data_ptr<int64_t>(),
        l1_late.data_ptr<bool>(),
        supp_ids.data_ptr<int64_t>(),
        lo_orderkeys.data_ptr<int64_t>(),
        lo_distinct_supp.data_ptr<int64_t>(),
        lo_late_distinct_supp.data_ptr<int64_t>(),
        reinterpret_cast<unsigned long long*>(counts.data_ptr<int64_t>()),
        n_rows,
        n_orders,
        num_suppliers
    );
    
    cudaDeviceSynchronize();
    
    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q21_launcher", &q21_launcher, "TPC-H Q21 supplier counting kernel");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q21_cuda_")
    cu_path = os.path.join(tmpdir, "_q21_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q21_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _launch_q21(
    l1_orderkey: torch.Tensor,
    l1_suppkey: torch.Tensor,
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    lo_orderkeys: torch.Tensor,
    lo_distinct_supp: torch.Tensor,
    lo_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
) -> torch.Tensor:
    return _EXT.q21_launcher(
        l1_orderkey,
        l1_suppkey,
        l1_late,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        num_suppliers,
    )

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
    """
    Fused CUDA kernel for TPC-H Q21 supplier filtering and counting.
    
    GPU execution: Single kernel pass performs binary search lookup,
    filter condition evaluation, and atomic counting per supplier.
    
    Why faster: Fuses searchsorted + masking + scatter_add into one kernel,
    reducing memory bandwidth usage and kernel launch overhead.
    Uses coalesced global memory access and one atomicAdd per qualifying thread.
    """
    counts = _launch_q21(
        l1_orderkey,
        l1_suppkey,
        l1_late,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        num_suppliers,
    )
    supp_idx = torch.arange(num_suppliers, device=counts.device)
    return supp_idx, counts

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
