import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void filter_group_sum_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int64_t* __restrict__ s_suppkey,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_lineitem,
    int64_t n_supplier,
    double* __restrict__ totals_out,
    int64_t* __restrict__ present_out
) {
    // Each block handles a chunk of lineitem rows and accumulates into supplier-local arrays in shared memory
    extern __shared__ double shared_totals[];
    int64_t* shared_present = (int64_t*)(shared_totals + BLOCK_SIZE);
    
    for (int i = threadIdx.x; i < BLOCK_SIZE; i++) {
        shared_totals[i] = 0.0;
        shared_present[i] = 0;
    }
    __syncthreads();

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_lineitem) {
        int32_t sd = l_shipdate_days[idx];
        if (sd >= date_lo && sd < date_hi) {
            int64_t sk = l_suppkey[idx];
            double rev = l_extendedprice[idx] * (1.0 - l_discount[idx]);
            
            // Find supplier index with binary search on sorted supplier keys
            // We'll do a simple linear search in shared memory after building a local cache
            // For simplicity and to avoid searchsorted overhead, use linear search within block
            // since supplier count is only 100k (fits in L2) but blocks are ~60k, each block has its own search
            // Better: pre-sort and do binary search on global memory with broadcast
            
            // Write to shared memory accumulators keyed by local supplier index (0..n_supplier-1)
            // But that would be too large for shared memory. Instead: each thread writes directly to
            // per-supplier accumulator in global memory, but with block-level reduction on supplier index.
            
            // Alternative approach: emit (suppkey, revenue) pairs, then use atomicAdd per supplier.
            // Since each lineitem row maps to one supplier, we can do atomicAdd directly.
            // Use a two-pass approach: first find which suppliers are present, then sum.
            
            // For this kernel we'll do a simpler approach: each thread writes to a per-block hash of suppliers
            // but because supplier count is large, we'll use global atomics per supplier
            
            // Actually, let's use a different strategy: since supplier count is 100k, each block can have
            // a shared memory array of 100k doubles -> not possible.
            
            // Best strategy: do block-level reduction on revenue, then one atomicAdd per unique supplier per block.
            // But each block may have many suppliers. Instead: use global atomicAdd per supplier.
            // This is memory-bound anyway, so atomics are fine if we use one per block per supplier.
            
            // For simplicity, each thread does atomicAdd directly to global totals array
            // Use a shared memory reduction per supplier index would be complex.
            // Given 60M rows and 100k suppliers, atomicAdd per thread is ~600 atoms/supplier on avg.
            // This can be slow. Better: use block-level hash map in shared memory for common suppliers.
            
            // Given time constraints, use direct global atomicAdd (will be correct)
            // But we can do block-level reduction first: each block accumulates into shared memory
            // for suppliers that appear in that block, then block does one atomicAdd per unique supplier.
            
            // Simple approach for correctness: atomicAdd per thread
            // First, need to map supplier key to index - use binary search on sorted supplier keys
            // Store sorted supplier keys in constant memory or global memory
            // For now, use a search on the fly (slow but correct)
        }
    }
}
"""

# Actually the above is getting too complex for a single kernel. Let's use a simpler approach:
# Do the filter and projection in CUDA, then use PyTorch scatter_add for the group-by.

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void filter_and_project_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n,
    int64_t* __restrict__ result_suppkey,
    double* __restrict__ result_revenue,
    int32_t* __restrict__ result_count
) {
    extern __shared__ int32_t shared_count[];
    int64_t* shared_suppkey = (int64_t*)(shared_count + blockDim.x);
    double* shared_revenue = (double*)(shared_suppkey + blockDim.x);

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadIdx.x < blockDim.x) {
        shared_count[threadIdx.x] = 0;
    }
    __syncthreads();

    if (idx < n) {
        int32_t sd = l_shipdate_days[idx];
        if (sd >= date_lo && sd < date_hi) {
            int64_t pos = atomicAdd(shared_count + threadIdx.x, 1);
            if (pos == 0) {
                shared_suppkey[threadIdx.x] = l_suppkey[idx];
                shared_revenue[threadIdx.x] = l_extendedprice[idx] * (1.0 - l_discount[idx]);
            } else {
                // Only keep first match per thread for simplicity
                // Better: use warp-level reduction
            }
        }
    }
    __syncthreads();

    // Write results - each thread writes at most one row if its count > 0
    if (threadIdx.x < blockDim.x && shared_count[threadIdx.x] > 0) {
        int32_t write_pos = atomicAdd(result_count, 1);
        result_suppkey[write_pos] = shared_suppkey[threadIdx.x];
        result_revenue[write_pos] = shared_revenue[threadIdx.x];
    }
}

// Simpler approach: just do filter and output all matching rows, then use PyTorch for group-by
__global__ void filter_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n,
    int64_t* __restrict__ out_suppkey,
    double* __restrict__ out_revenue,
    int32_t* __restrict__ out_count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t sd = l_shipdate_days[idx];
    bool keep = (sd >= date_lo) && (sd < date_hi有趣)  ;  // syntax error forcompletion
    
    // Actually let's just do it correctly
}
"""

# Given the complexity of doing group-by in CUDA, let's use a simpler approach:
# Do filter in CUDA (fast fused kernel), then group-by using PyTorch scatter_add (efficient GPU op)

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void filter_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n,
    int64_t* __restrict__ out_suppkey,
    double* __restrict__ out_revenue,
    int32_t* __restrict__ out_count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int32_t sd = l_shipdate_days[idx];
    if (sd >= date_lo && sd < date_hi) {
        int64_t sk = l_suppkey[idx];
        double rev = l_extendedprice[idx] * (1.0 - l_discount[idx]);
        int32_t pos = atomicAdd(out_count, 1);
        out_suppkey[pos] = sk;
        out_revenue[pos] = rev;
    }
}

int filter_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    int32_t date_lo,
    int32_t date_hi,
    torch::Tensor out_suppkey,
    torch::Tensor out_revenue,
    torch::Tensor out_count
) {
    auto n = l_suppkey.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    filter_kernel<<<blocks, BLOCK_SIZE>>>(
        l_suppkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate_days.data_ptr<int32_t>(),
        date_lo, date_hi, n,
        out_suppkey.data_ptr<int64_t>(),
        out_revenue.data_ptr<double>(),
        out_count.data_ptr<int32_t>()
    );
    // Return the count so we can resize
    int32_t count_h = 0;
    cudaMemcpy(&count_h, out_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return (int)count_h;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_launcher", &filter_launcher, "Filter lineitem rows by shipdate window and compute projected revenue");
}
"""


def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q15_cuda_")
    cu_path = os.path.join(tmpdir, "_q15_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q15_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )


_EXT = _load_cuda()


def _query_core(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated core for TPC-H Q15.
    
    Approach:
    1. Use a single fused CUDA kernel to filter lineitem rows by shipdate window
       and compute projected revenue (price * (1 - discount)) in one pass.
       This replaces the PyTorch mask + indexing + multiplication steps.
    2. Use PyTorch scatter_add for group-by aggregation (efficient GPU scatter operation).
    3. Use PyTorch for finding max revenue and selecting top suppliers.
    
    Why faster: Fused filter+project in CUDA eliminates intermediate mask tensor and
    avoids two extra kernel launches (mask creation, indexing). The rest of the pipeline
    uses efficient PyTorch scatter_add which is bandwidth-optimal for the group-by.
    """
    n = l_suppkey.numel()
    device = l_suppkey.device
    
    # Allocate output buffers (conservative max = all rows pass filter)
    out_suppkey = torch.empty(n, dtype=torch.int64, device=device)
    out_revenue = torch.empty(n, dtype=torch.float64, device=device)
    out_count = torch.zeros(1, dtype=torch.int32, device=device)
    
    # Run fused filter kernel
    count = _EXT.filter_launcher(
        l_suppkey, l_extendedprice, l_discount, l_shipdate_days,
        shipdate_lo_days, shipdate_hi_days,
        out_suppkey, out_revenue, out_count
    )
    
    if count == 0:
        # No matching rows - return empty result consistent with reference
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
        )
    
    # Trim to actual count
    out_suppkey = out_suppkey[:count]
    out_revenue = out_revenue[:count]
    
    # Group-by supplier using scatter_add
    # Map each row to supplier index via searchsorted (supplier keys are unique)
    sorted_s, sort_idx = torch.sort(s_suppkey)
    pos = torch.searchsorted(sorted_s, out_suppkey)
    
    # Guard: only keep rows whose supplier exists in supplier table
    valid = (pos < sorted_s.numel()) & (sorted_s[pos.clamp(max=sorted_s.numel() - 1)] == out_suppkey)
    pos = pos[valid]
    out_revenue = out_revenue[valid]
    
    n_sup = s_suppkey.numel()
    totals_sorted = torch.zeros(n_sup, dtype=torch.float64, device=device)
    totals_sorted.scatter_add_(0, pos, out_revenue)
    
    # Unsort back to original s_suppkey order
    totals = torch.empty_like(totals_sorted)
    totals[sort_idx] = totals_sorted
    
    # Track which suppliers had entries
    present_sorted = torch.zeros(n_sup, dtype=torch.bool, device=device)
    present_sorted.scatter_(0, pos, torch.ones_like(pos, dtype=torch.bool))
    present = torch.empty_like(present_sorted)
    present[sort_idx] = present_sorted
    
    # Max revenue over present suppliers
    max_rev = totals[present].max()
    
    # Select suppliers equal to max revenue
    sel_mask = present & (totals == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    
    # Order by s_suppkey
    sel_keys = s_suppkey[sel_idx]
    order = torch.argsort(sel_keys)
    sel_idx = sel_idx[order]
    
    return sel_idx, s_suppkey[sel_idx], totals[sel_idx]


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1996-01-01",
    shipdate_hi: str = "1996-04-01",
) -> cudf.DataFrame:
    """
    SQL:
      WITH revenue AS (
        SELECT l_suppkey AS supplier_no,
               SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
        GROUP BY l_suppkey
      )
      SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
      FROM supplier, revenue
      WHERE s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
      ORDER BY s_suppkey;
    Description: TPC-H Q15 — find supplier(s) with the top revenue in a given quarter.
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_price_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_disc_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)

    sel_idx_t, sel_keys_t, sel_rev_t = _query_core(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        s_suppkey_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # Use indices to gather string/phone columns from cuDF supplier table
    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_t),
    })
