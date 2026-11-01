import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <algorithm>

#define BLOCK_SIZE 1024

// Binary search for orderkey index
__device__ __forceinline__ int64_t find_order_idx(
    const int64_t* __restrict__ orderkeys,
    int64_t n_orders,
    int64_t target
) {
    int64_t lo = 0, hi = n_orders - 1;
    while (lo <= hi) {
        int64_t mid = (lo + hi) >> 1;
        if (orderkeys[mid] < target) {
            lo = mid + 1;
        } else if (orderkeys[mid] > target) {
            hi = mid - 1;
        } else {
            return mid;
        }
    }
    return lo;
}

__global__ void q21_fused_kernel(
    const int64_t* __restrict__ l_orderkeys,
    const int8_t*  __restrict__ l_late,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int32_t* __restrict__ lo_distinct_supp,
    const int32_t* __restrict__ lo_late_distinct_supp,
    int64_t n_l1,
    int64_t n_orders,
    unsigned long long* __restrict__ out_counts
) {
    int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n_l1) {
        int64_t my_orderkey = l_orderkeys[gid];
        int64_t idx = find_order_idx(lo_orderkeys, n_orders, my_orderkey);
        
        if (idx < n_orders) {
            int64_t distinct_supp = lo_distinct_supp[idx];
            int64_t late_distinct_supp = lo_late_distinct_supp[idx];
            
            bool is_late = l_late[gid] != 0;
            bool exists_other = distinct_supp >= 2;
            bool not_exists_other_late = late_distinct_supp <= 1;
            int64_t my_supp_id = supp_ids[gid];
            
            if (is_late && exists_other && not_exists_other_late && my_supp_id >= 0) {
                atomicAdd(&out_counts[my_supp_id], 1ULL);
            }
        }
    }
}

torch::Tensor q21_launcher(
    torch::Tensor l_orderkeys,
    torch::Tensor l_late,
    torch::Tensor supp_ids,
    torch::Tensor lo_orderkeys,
    torch::Tensor lo_distinct_supp,
    torch::Tensor lo_late_distinct_supp,
    int64_t num_suppliers
) {
    TORCH_CHECK(l_orderkeys.scalar_type() == torch::kInt64);
    TORCH_CHECK(l_late.scalar_type() == torch::kInt8);
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64);
    TORCH_CHECK(lo_orderkeys.scalar_type() == torch::kInt64);
    TORCH_CHECK(lo_distinct_supp.scalar_type() == torch::kInt32);
    TORCH_CHECK(lo_late_distinct_supp.scalar_type() == torch::kInt32);
    
    auto move_to_cuda = [](torch::Tensor t) {
        if (!t.is_cuda()) return t.cuda();
        return t;
    };
    l_orderkeys = move_to_cuda(l_orderkeys).contiguous();
    l_late = move_to_cuda(l_late).contiguous();
    supp_ids = move_to_cuda(supp_ids).contiguous();
    lo_orderkeys = move_to_cuda(lo_orderkeys).contiguous();
    lo_distinct_supp = move_to_cuda(lo_distinct_supp).contiguous();
    lo_late_distinct_supp = move_to_cuda(lo_late_distinct_supp).contiguous();
    
    int64_t n_l1 = l_orderkeys.numel();
    int64_t n_orders = lo_orderkeys.numel();
    
    // Use unsigned long long for atomicAdd compatibility
    auto out_counts = torch::zeros({num_suppliers}, l_orderkeys.options().dtype(torch::kInt64)).to(torch::kInt64);
    unsigned long long* out_ptr = (unsigned long long*)out_counts.data_ptr<int64_t>();
    
    int blocks = (n_l1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q21_fused_kernel<<<blocks, BLOCK_SIZE>>>(
        l_orderkeys.data_ptr<int64_t>(),
        l_late.data_ptr<int8_t>(),
        supp_ids.data_ptr<int64_t>(),
        lo_orderkeys.data_ptr<int64_t>(),
        lo_distinct_supp.data_ptr<int32_t>(),
        lo_late_distinct_supp.data_ptr<int32_t>(),
        n_l1,
        n_orders,
        out_ptr
    );
    
    return out_counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q21_launcher", &q21_launcher, "TPC-H Q21 fused kernel (CUDA)");
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
    CUDA-accelerated core: fused filter + aggregate.
    Replaces the reference's searchsorted + scatter_add with a single kernel
    that performs binary search, predicate evaluation, and atomic add per qualifying row.
    """
    l_late_int8 = l1_late.to(torch.int8)
    
    counts = _EXT.q21_launcher(
        l1_orderkey,
        l_late_int8,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        num_suppliers
    )
    
    supp_idx = torch.arange(num_suppliers, device=counts.device, dtype=torch.int64)
    return supp_idx, counts


def run_query(tables: dict[str, cudf.DataFrame], nation_name: str = "SAUDI ARABIA", orderstatus: str = "F") -> cudf.DataFrame:
    """
    SQL: TPC-H Q21 - Suppliers Who Kept Orders Waiting.
    Description: For each supplier in the given nation, count orders of status 'F' where the
                 supplier's lineitem was late, another supplier also shipped the order, and no
                 other supplier was late on that order.
    
    Implementation:
      - cuDF: joins, filters, per-orderkey aggregates (distinct suppliers, distinct late suppliers).
      - CUDA fused kernel: single pass over l1 rows performing binary search on orderkey index,
        predicate evaluation, and atomic increment on supplier counts.
    
    Why faster:
      - Single fused kernel replaces searchsorted + mask + scatter_add
      - Coalesced memory access for all columns
      - No intermediate tensors; direct atomic add to final output
    
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
    lo_distinct_supp_t = torch.as_tensor(agg["distinct_supp"].to_cupy(), dtype=torch.int32)
    lo_late_distinct_supp_t = torch.as_tensor(agg["late_distinct_supp"].to_cupy(), dtype=torch.int32)

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
        "s_name": supp_n["s_name"].iloc[cudf.Series(supp_idx_t.cpu().numpy())].reset_index(drop=True),
        "numwait": cudf.Series(counts_t.cpu().numpy()),
    })
    result = result[result["numwait"] > 0]
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100).reset_index(drop=True)
    return result
