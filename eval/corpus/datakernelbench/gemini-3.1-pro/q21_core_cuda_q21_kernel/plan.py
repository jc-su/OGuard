import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 512

__global__ void q21_kernel(
    const int64_t* __restrict__ l1_orderkey,
    const bool* __restrict__ l1_late,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int64_t* __restrict__ lo_distinct_supp,
    const int64_t* __restrict__ lo_late_distinct_supp,
    unsigned long long int* __restrict__ counts,
    int64_t num_l1,
    int64_t num_lo
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_l1) return;

    // Filter early to avoid expensive binary search when possible
    if (!l1_late[idx]) return;
    int64_t s_id = supp_ids[idx];
    if (s_id < 0) return;

    int64_t okey = l1_orderkey[idx];
    
    // Binary search for the exact orderkey in the unique sorted lo_orderkeys array
    int64_t left = 0;
    int64_t right = num_lo;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        if (lo_orderkeys[mid] < okey) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    // Ensure found and retrieve aggregated counts
    if (left < num_lo) {
        int64_t d_supp = lo_distinct_supp[left];
        int64_t l_d_supp = lo_late_distinct_supp[left];
        
        // EXISTS other supplier (distinct_supp >= 2)
        // NOT EXISTS other late supplier (late_distinct_supp <= 1)
        if (d_supp >= 2 && l_d_supp <= 1) {
            atomicAdd(counts + s_id, 1ULL);
        }
    }
}

std::vector<torch::Tensor> q21_launcher(
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
    TORCH_CHECK(lo_orderkeys.scalar_type() == torch::kInt64);
    
    if (!l1_orderkey.is_cuda()) l1_orderkey = l1_orderkey.cuda();
    if (!l1_late.is_cuda()) l1_late = l1_late.cuda();
    if (!supp_ids.is_cuda()) supp_ids = supp_ids.cuda();
    if (!lo_orderkeys.is_cuda()) lo_orderkeys = lo_orderkeys.cuda();
    if (!lo_distinct_supp.is_cuda()) lo_distinct_supp = lo_distinct_supp.cuda();
    if (!lo_late_distinct_supp.is_cuda()) lo_late_distinct_supp = lo_late_distinct_supp.cuda();

    l1_orderkey = l1_orderkey.contiguous();
    l1_late = l1_late.contiguous();
    supp_ids = supp_ids.contiguous();
    lo_orderkeys = lo_orderkeys.contiguous();
    lo_distinct_supp = lo_distinct_supp.contiguous();
    lo_late_distinct_supp = lo_late_distinct_supp.contiguous();

    int64_t num_l1 = l1_orderkey.numel();
    int64_t num_lo = lo_orderkeys.numel();

    auto counts = torch::zeros({num_suppliers}, torch::dtype(torch::kInt64).device(l1_orderkey.device()));
    
    if (num_l1 > 0) {
        int blocks = (num_l1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        q21_kernel<<<blocks, BLOCK_SIZE>>>(
            l1_orderkey.data_ptr<int64_t>(),
            (const bool*)l1_late.data_ptr<bool>(),
            supp_ids.data_ptr<int64_t>(),
            lo_orderkeys.data_ptr<int64_t>(),
            lo_distinct_supp.data_ptr<int64_t>(),
            lo_late_distinct_supp.data_ptr<int64_t>(),
            (unsigned long long int*)counts.data_ptr<int64_t>(),
            num_l1,
            num_lo
        );
    }
    
    auto supp_idx = torch::arange(num_suppliers, torch::dtype(torch::kInt64).device(l1_orderkey.device()));
    return {supp_idx, counts};
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
    """
    Executes the fused Q21 evaluation kernel.
    Replaces torch.searchsorted + gather + scatter_add with a single C++ kernel
    that skips search altogether for rows filtered by `!l1_late`.
    """
    res = _EXT.q21_launcher(
        l1_orderkey,
        l1_suppkey,
        l1_late,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        num_suppliers
    )
    return res[0], res[1]

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
