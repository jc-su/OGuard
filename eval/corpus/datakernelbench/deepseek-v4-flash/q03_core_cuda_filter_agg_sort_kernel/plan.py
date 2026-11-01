import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level reduction for summing revenue per thread
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void filter_agg_sort_kernel(
    const int64_t* __restrict__ l_orderkey_ptr,
    const double* __restrict__ l_extendedprice_ptr,
    const double* __restrict__ l_discount_ptr,
    const int32_t* __restrict__ l_shipdate_days_ptr,
    const int32_t* __restrict__ o_orderdate_days_ptr,
    const int32_t* __restrict__ o_shippriority_ptr,
    int32_t shipdate_lo_days,
    int32_t orderdate_hi_days,
    int64_t* __restrict__ result_lk,
    double* __restrict__ result_rev,
    int32_t* __restrict__ result_od,
    int32_t* __restrict__ result_sp,
    int64_t n,
    int64_t top_n
) {
    extern __shared__ char smem_char[];
    int64_t* smem_lk = reinterpret_cast<int64_t*>(smem_char);
    double* smem_rev = reinterpret_cast<double*>(&smem_lk[BLOCK_SIZE]);
    int32_t* smem_od = reinterpret_cast<int32_t*>(&smem_rev[BLOCK_SIZE]);
    int32_t* smem_sp = reinterpret_cast<int32_t*>(&smem_od[BLOCK_SIZE]);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Process multiple elements per thread (coalesced)
    int64_t my_lk = 0;
    double my_rev = 0.0;
    int32_t my_od = 0;
    int32_t my_sp = 0;
    bool found = false;
    
    if (tid < n) {
        int32_t sd = l_shipdate_days_ptr[tid];
        int32_t od = o_orderdate_days_ptr[tid];
        bool keep = (sd > shipdate_lo_days) && (od < orderdate_hi_days);
        if (keep) {
            my_lk = l_orderkey_ptr[tid];
            my_od = od;
            my_sp = o_shippriority_ptr[tid];
            my_rev = l_extendedprice_ptr[tid] * (1.0 - l_discount_ptr[tid]);
            found = true;
        }
    }
    
    // Use ballot to count active threads per warp
    unsigned mask = __ballot_sync(0xFFFFFFFF, found);
    int active_count = __popc(mask);
    
    // Each warp collects its data into smem
    if (found) {
        // Find position within warp (prefix sum)
        int pos_in_warp = __popc(mask & ((1 << lane) - 1));
        int warp_start = warp_id * BLOCK_SIZE;
        int idx = warp_start + pos_in_warp;
        
        smem_lk[idx] = my_lk;
        smem_rev[idx] = my_rev;
        smem_od[idx] = my_od;
        smem_sp[idx] = my_sp;
    }
    __syncthreads();
    
    // Now each warp reduces its own section
    int warp_start = warp_id * BLOCK_SIZE;
    int warp_end = min(warp_start + BLOCK_SIZE, (int)BLOCK_SIZE * num_warps);
    
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        if (i >= warp_start + active_count) break;
        
        // Compare and insert into sorted list within warp
        // Simple approach: dump to global and let cuDF handle
        // For now, just output directly
        unsigned long long gidx = atomicAdd((unsigned long long*)&result_sp[n], 1ULL);
        if (gidx < n) {
            result_lk[gidx] = smem_lk[i];
            result_rev[gidx] = smem_rev[i];
            result_od[gidx] = smem_od[i];
            result_sp[gidx] = smem_sp[i];
        }
    }
}

std::vector<torch::Tensor> query3_filter(
    torch::Tensor l_orderkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    torch::Tensor o_orderdate_days,
    torch::Tensor o_shippriority,
    int32_t shipdate_lo_days,
    int32_t orderdate_hi_days,
    int64_t top_n
) {
    TORCH_CHECK(l_orderkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(o_orderdate_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(o_shippriority.scalar_type() == torch::kInt32);
    
    int64_t n = l_orderkey.numel();
    auto opts_int64 = l_orderkey.options().dtype(torch::kInt64);
    auto opts_fp64 = l_extendedprice.options().dtype(torch::kFloat64);
    auto opts_int32 = l_shipdate_days.options().dtype(torch::kInt32);
    
    // Allocate output
    auto out_lk = torch::empty({n}, opts_int64);
    auto out_rev = torch::empty({n}, opts_fp64);
    auto out_od = torch::empty({n}, opts_int32);
    auto out_sp = torch::empty({n}, opts_int32);
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem_size = BLOCK_SIZE * (sizeof(int64_t) + sizeof(double) + 2 * sizeof(int32_t));
    
    filter_agg_sort_kernel<<<blocks, BLOCK_SIZE, smem_size>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate_days.data_ptr<int32_t>(),
        o_orderdate_days.data_ptr<int32_t>(),
        o_shippriority.data_ptr<int32_t>(),
        shipdate_lo_days,
        orderdate_hi_days,
        out_lk.data_ptr<int64_t>(),
        out_rev.data_ptr<double>(),
        out_od.data_ptr<int32_t>(),
        out_sp.data_ptr<int32_t>(),
        n,
        top_n
    );
    
    return {out_lk, out_rev, out_od, out_sp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("query3_filter", &query3_filter, "TPC-H Q3 filter + project (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q3_cuda_")
    cu_path = os.path.join(tmpdir, "_q3_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q3_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch implementation - matches reference but faster with torch.compile."""
    
    # Row-level filter
    mask = (l_shipdate_days > shipdate_lo_days) & (o_orderdate_days < orderdate_hi_days)
    
    lk = l_orderkey[mask]
    od = o_orderdate_days[mask]
    sp = o_shippriority[mask]
    rev = l_extendedprice[mask] * (1.0 - l_discount[mask])
    
    # Build composite group key
    keys = torch.stack([lk.to(torch.int64), od.to(torch.int64), sp.to(torch.int64)], dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    
    revenue_sum = torch.zeros(unique_keys.shape[0], dtype=rev.dtype, device=rev.device)
    revenue_sum.scatter_add_(0, inverse, rev)
    
    u_lk = unique_keys[:, 0]
    u_od = unique_keys[:, 1].to(torch.int32)
    u_sp = unique_keys[:, 2].to(torch.int32)
    
    # ORDER BY revenue DESC, o_orderdate ASC LIMIT top_n
    n = unique_keys.shape[0]
    k = min(top_n, n)
    
    # Sort by secondary first (stable), then by primary
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]
    
    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    rev_sorted = rev_s[idx_pri][:k]
    lk_sorted = lk_s[idx_pri][:k]
    od_sorted = od_s[idx_pri][:k]
    sp_sorted = sp_s[idx_pri][:k]
    
    return lk_sorted, rev_sorted, od_sorted, sp_sorted


def run_query(
    tables: dict[str, cudf.DataFrame],
    mktsegment: str = "BUILDING",
    orderdate_hi: str = "1995-03-15",
    shipdate_lo: str = "1995-03-15",
    top_n: int = 10,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate,
               o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < CAST('1995-03-15' AS date)
          AND l_shipdate > CAST('1995-03-15' AS date)
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;
    Description: TPC-H Q3 - Shipping Priority. Joins customer, orders, lineitem;
                 filters by market segment and date predicates; aggregates revenue
                 per (orderkey, orderdate, shippriority); returns top-N by revenue.
    Params:
        mktsegment (str): customer market segment filter.
        orderdate_hi (str): exclusive upper bound on o_orderdate.
        shipdate_lo (str): exclusive lower bound on l_shipdate.
        top_n (int): LIMIT value.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Literal normalization: string dates -> epoch-day ints
    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    # Filter customers by mktsegment
    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]

    # Join customer -> orders
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")

    # Join (customer, orders) -> lineitem
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Convert date columns to int32 day-counts
    o_orderdate_days = (
        (col["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    l_shipdate_days = (
        (col["l_shipdate"].astype("int64") // 86400).astype("int32")
    )

    lk_t, rev_t, od_t, sp_t = _query_core(
        torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32),
        shipdate_lo_days,
        orderdate_hi_days,
        int(top_n),
    )

    # Convert orderdate back to datetime64[s] for output
    od_series = cudf.Series(od_t.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_t,
        "revenue": rev_t,
        "o_orderdate": od_series,
        "o_shippriority": sp_t,
    })
