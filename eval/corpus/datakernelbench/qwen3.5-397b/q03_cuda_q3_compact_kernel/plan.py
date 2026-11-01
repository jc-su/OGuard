import os
import tempfile
import torch
import cudf
import pandas as pd
import numpy as np
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Fused kernel: filter, project, and write valid rows with correct global indices
__global__ void q3_compact_kernel(
    const int64_t* l_orderkey,
    const double* l_price,
    const double* l_discount,
    const int32_t* l_shipdate,
    const int32_t* o_orderdate,
    const int32_t* o_shippriority,
    int32_t shipdate_lo,
    int32_t orderdate_hi,
    const int* block_prefix_sums,
    int64_t* out_lk,
    double* out_rev,
    int32_t* out_od,
    int32_t* out_sp,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    bool is_valid = (l_shipdate[idx] > shipdate_lo && o_orderdate[idx] < orderdate_hi);
    
    __shared__ int s_valid[BLOCK_SIZE];
    s_valid[threadIdx.x] = is_valid ? 1 : 0;
    __syncthreads();
    
    // Inclusive prefix sum within block
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int val = 0;
        if (threadIdx.x >= stride) {
            val = s_valid[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            s_valid[threadIdx.x] += val;
        }
        __syncthreads();
    }
    
    if (is_valid) {
        // Global write index = prefix sum of all previous blocks + local rank
        int local_rank = s_valid[threadIdx.x] - 1;  // 0-indexed within block
        int64_t write_idx = (int64_t)block_prefix_sums[blockIdx.x] + local_rank;
        
        out_lk[write_idx] = l_orderkey[idx];
        out_od[write_idx] = o_orderdate[idx];
        out_sp[write_idx] = o_shippriority[idx];
        out_rev[write_idx] = l_price[idx] * (1.0 - l_discount[idx]);
    }
}

// Kernel to compute per-block valid counts and prefix sums
__global__ void compute_block_prefix_kernel(
    const int32_t* l_shipdate,
    const int32_t* o_orderdate,
    int32_t shipdate_lo,
    int32_t orderdate_hi,
    int* block_counts,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int valid = 0;
    if (idx < n) {
        if (l_shipdate[idx] > shipdate_lo && o_orderdate[idx] < orderdate_hi) {
            valid = 1;
        }
    }
    
    __shared__ int s_count[BLOCK_SIZE];
    s_count[threadIdx.x] = valid;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_count[threadIdx.x] += s_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = s_count[0];
    }
}

std::vector<torch::Tensor> q3_launcher(
    torch::Tensor l_orderkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    torch::Tensor o_orderdate_days,
    torch::Tensor o_shippriority,
    int32_t shipdate_lo,
    int32_t orderdate_hi
) {
    // Ensure CUDA and contiguous
    if (!l_orderkey.is_cuda()) l_orderkey = l_orderkey.cuda();
    if (!l_extendedprice.is_cuda()) l_extendedprice = l_extendedprice.cuda();
    if (!l_discount.is_cuda()) l_discount = l_discount.cuda();
    if (!l_shipdate_days.is_cuda()) l_shipdate_days = l_shipdate_days.cuda();
    if (!o_orderdate_days.is_cuda()) o_orderdate_days = o_orderdate_days.cuda();
    if (!o_shippriority.is_cuda()) o_shippriority = o_shippriority.cuda();
    
    l_orderkey = l_orderkey.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();
    o_orderdate_days = o_orderdate_days.contiguous();
    o_shippriority = o_shippriority.contiguous();
    
    int64_t n = l_orderkey.numel();
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate block_counts on device
    auto block_counts = torch::empty({num_blocks}, l_orderkey.options().dtype(torch::kInt32));
    
    // Launch count kernel
    compute_block_prefix_kernel<<<num_blocks, BLOCK_SIZE>>>(
        l_shipdate_days.data_ptr<int32_t>(),
        o_orderdate_days.data_ptr<int32_t>(),
        shipdate_lo,
        orderdate_hi,
        block_counts.data_ptr<int>(),
        n
    );
    
    // Copy counts to host to compute prefix sums (block start offsets)
    auto counts_cpu = block_counts.cpu();
    auto counts_data = counts_cpu.data_ptr<int>();
    
    // Compute prefix sums (start offsets for each block)
    std::vector<int> prefix_sums(num_blocks);
    int current_offset = 0;
    int total_valid = 0;
    for (int i = 0; i < num_blocks; ++i) {
        prefix_sums[i] = current_offset;
        current_offset += counts_data[i];
    }
    total_valid = current_offset;
    
    // Copy prefix sums to device
    auto block_prefix_sums = torch::tensor(prefix_sums, l_orderkey.options().dtype(torch::kInt32));
    
    // Allocate output buffers
    auto out_lk = torch::empty({total_valid > 0 ? total_valid : 1}, l_orderkey.options());
    auto out_rev = torch::empty({total_valid > 0 ? total_valid : 1}, l_extendedprice.options());
    auto out_od = torch::empty({total_valid > 0 ? total_valid : 1}, o_orderdate_days.options());
    auto out_sp = torch::empty({total_valid > 0 ? total_valid : 1}, o_shippriority.options());
    
    // Launch compact kernel
    q3_compact_kernel<<<num_blocks, BLOCK_SIZE>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate_days.data_ptr<int32_t>(),
        o_orderdate_days.data_ptr<int32_t>(),
        o_shippriority.data_ptr<int32_t>(),
        shipdate_lo,
        orderdate_hi,
        block_prefix_sums.data_ptr<int>(),
        out_lk.data_ptr<int64_t>(),
        out_rev.data_ptr<double>(),
        out_od.data_ptr<int32_t>(),
        out_sp.data_ptr<int32_t>(),
        n
    );
    
    // Slice outputs to valid count (handle empty case)
    if (total_valid == 0) {
        return {
            out_lk.slice(0, 0, 0),
            out_rev.slice(0, 0, 0),
            out_od.slice(0, 0, 0),
            out_sp.slice(0, 0, 0)
        };
    }
    
    return {
        out_lk.slice(0, 0, total_valid),
        out_rev.slice(0, 0, total_valid),
        out_od.slice(0, 0, total_valid),
        out_sp.slice(0, 0, total_valid)
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q3_launcher", &q3_launcher, "TPC-H Q3 CUDA Compaction");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q3_cuda_")
    cu_path = os.path.join(tmpdir, "_q3_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    
    # H100 is sm_90
    return load(
        name="q3_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load extension at module import
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
    """
    Optimized core using CUDA for filter/project/compaction.
    PyTorch handles grouping (unique + scatter) and sorting on compacted data.
    """
    # CUDA Compaction: Filter + Project + Write Valid Rows
    # Returns sliced tensors containing only valid rows
    lk_comp, rev_comp, od_comp, sp_comp = _EXT.q3_launcher(
        l_orderkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        o_orderdate_days,
        o_shippriority,
        shipdate_lo_days,
        orderdate_hi_days,
    )
    
    n_valid = lk_comp.shape[0]
    if n_valid == 0:
        # Return empty tensors with correct dtypes
        return (
            lk_comp.to(torch.int64),
            rev_comp.to(torch.float64),
            od_comp.to(torch.int32),
            sp_comp.to(torch.int32),
        )

    # GroupBy: (l_orderkey, o_orderdate, o_shippriority) -> Sum(revenue)
    # Stack keys for unique
    keys = torch.stack([lk_comp.to(torch.int64), od_comp.to(torch.int64), sp_comp.to(torch.int64)], dim=1)
    
    # unique returns sorted unique rows by default
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    
    revenue_sum = torch.zeros(unique_keys.shape[0], dtype=rev_comp.dtype, device=rev_comp.device)
    revenue_sum.scatter_add_(0, inverse, rev_comp)
    
    u_lk = unique_keys[:, 0]
    u_od = unique_keys[:, 1].to(torch.int32)
    u_sp = unique_keys[:, 2].to(torch.int32)
    
    # Sort: revenue DESC, o_orderdate ASC
    # Stable sort by secondary key first
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]
    
    # Then by primary key (revenue)
    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    
    k = min(top_n, idx_pri.shape[0])
    
    return (
        lk_s[idx_pri][:k],
        rev_s[idx_pri][:k],
        od_s[idx_pri][:k],
        sp_s[idx_pri][:k],
    )


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
    Description: TPC-H Q3 - Shipping Priority. 
                 Uses CUDA for filter/project/compaction to reduce data volume before grouping.
                 Grouping and sorting performed in PyTorch on compacted data.
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
