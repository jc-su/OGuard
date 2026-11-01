import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cstdint>

#define BLOCK_SIZE 1024
#define NUM_GROUPS 1900  // Max possible years (covers TPC-H scale)

__global__ void q8_kernel(
    const int32_t* __restrict__ o_year_ptr,
    const double* __restrict__ volume_ptr,
    const bool* __restrict__ is_brazil_ptr,
    int32_t* __restrict__ unique_years_out,
    double* __restrict__ sum_brazil_out,
    double* __restrict__ sum_total_out,
    int32_t* __restrict__ unique_years_count,
    int64_t n_elements
) {
    // Each block handles a range of elements using shared memory hash into small array
    __shared__ int32_t s_years[BLOCK_SIZE];
    __shared__ double s_brazil[BLOCK_SIZE];
    __shared__ double s_total[BLOCK_SIZE];
    __shared__ int32_t s_count;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        s_count = 0;
    }
    __syncthreads();

    double my_brazil = 0.0;
    double my_total = 0.0;
    int32_t my_year = 0;

    if (idx < n_elements) {
        my_year = o_year_ptr[idx];
        my_total = volume_ptr[idx];
        my_brazil = is_brazil_ptr[idx] ? my_total : 0.0;
    }

    // Simple shared memory accumulation for this block (handles up to ~1900 unique years)
    // We'll just accumulate in local variables and use atomicAdd per block
    __shared__ double block_brazil;
    __shared__ double block_total;
    if (threadIdx.x == 0) {
        block_brazil = 0.0;
        block_total = 0.0;
    }
    __syncthreads();

    // Block-level reduction
    __shared__ double red_brazil[BLOCK_SIZE];
    __shared__ double red_total[BLOCK_SIZE];
    
    red_brazil[threadIdx.x] = my_brazil;
    red_total[threadIdx.x] = my_total;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red_brazil[threadIdx.x] += red_brazil[threadIdx.x + s];
            red_total[threadIdx.x] += red_total[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Store all years from this block and atomics to global arrays
        // Since we need years as well, we'll do a global atomic gather
        atomicAdd(sum_brazil_out, red_brazil[0]);
        atomicAdd(sum_total_out, red_total[0]);
    }
}

__global__ void q8_per_year_kernel(
    const int32_t* __restrict__ o_year_ptr,
    const double* __restrict__ volume_ptr,
    const bool* __restrict__ is_brazil_ptr,
    int32_t* __restrict__ year_keys_out,
    double* __restrict__ sum_brazil_out,
    double* __restrict__ sum_total_out,
    int32_t* __restrict__ num_unique_years,
    int64_t n_elements
) {
    // This kernel uses atomic per unique year - suitable when unique years is small (max ~6 for TPC-H Q8)
    extern __shared__ int32_t s_keys[];
    double* s_brazil = (double*)&s_keys[BLOCK_SIZE];
    double* s_total = (double*)&s_brazil[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory (first thread only, but we need per-block init)
    // Use grid stride loop approach instead for correctness
    if (threadIdx.x == 0) {
        *num_unique_years = 0;
    }
    __syncthreads();

    // Grid-stride loop to handle all elements
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += gridDim.x * blockDim.x) {
        int32_t yr = o_year_ptr[i];
        double vol = volume_ptr[i];
        double brazil_vol = is_brazil_ptr[i] ? vol : 0.0;
        
        // Atomic accumulate to global arrays (per-year aggregates)
        // Since years are small range we use direct atomicAdd on global output
        // But first we need to know where this year maps to in output
        // We'll use a simpler approach: global atomicAdd to dedicated per-year slots
        // Year range is small (1992-1998 for TPC-H), so we can use offset
        // For robustness, we'll scatter into a larger array and then compact
        atomicAdd(&sum_brazil_out[yr - 1990], brazil_vol);
        atomicAdd(&sum_total_out[yr - 1990], vol);
        // Mark year as present
        if (threadIdx.x == 0) {
            // We need a separate flag array, let's use the years array
            // Actually let's just use atomicMax to track range
        }
    }
}

// Second kernel to compact the results (since year range is small, we can just iterate)
__global__ void compact_kernel(
    double* __restrict__ sum_brazil_arr,
    double* __restrict__ sum_total_arr,
    int32_t* __restrict__ year_keys_out,
    double* __restrict__ mkt_share_out,
    int32_t* __restrict__ num_years_out,
    int32_t min_year,
    int32_t max_year
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int year = min_year + tid;
    
    if (year <= max_year) {
        int idx = year - 1990;
        double total = sum_total_arr[idx];
        if (total > 0.0) {
            int pos = atomicAdd(num_years_out, 1);
            year_keys_out[pos] = year;
            mkt_share_out[pos] = sum_brazil_arr[idx] / total;
        }
    }
}

// Optimized single-kernel approach: first pass counts per year, second pass aggregates
// For TPC-H Q8, we know years are in small range, use atomic per year
void q8_launcher(
    torch::Tensor o_year,
    torch::Tensor volume,
    torch::Tensor is_brazil,
    torch::Tensor out_years,
    torch::Tensor out_mkt_share
) {
    TORCH_CHECK(o_year.scalar_type() == torch::kInt32);
    TORCH_CHECK(volume.scalar_type() == torch::kFloat64);
    TORCH_CHECK(is_brazil.scalar_type() == torch::kBool);
    
    if (!o_year.is_cuda()) o_year = o_year.cuda();
    if (!volume.is_cuda()) volume = volume.cuda();
    if (!is_brazil.is_cuda()) is_brazil = is_brazil.cuda();
    
    o_year = o_year.contiguous();
    volume = volume.contiguous();
    is_brazil = is_brazil.contiguous();
    
    int64_t n = o_year.numel();
    int64_t max_years = 10; // TPC-H Q8 has at most 6 years (1992-1998)
    
    // Allocate temporary arrays for per-year accumulation (max 600 years to be safe, but we know it's small)
    auto sum_brazil_arr = torch::zeros({max_years}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto sum_total_arr = torch::zeros({max_years}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = min(blocks, 2048); // Cap blocks
    
    q8_per_year_kernel<<<blocks, BLOCK_SIZE, 0>>>(  // No dynamic shared memory needed
        o_year.data_ptr<int32_t>(),
        volume.data_ptr<double>(),
        is_brazil.data_ptr<bool>(),
        nullptr, // not used in this kernel
        sum_brazil_arr.data_ptr<double>(),
        sum_total_arr.data_ptr<double>(),
        nullptr,
        n
    );
    
    // Find min/max years present by scanning a sample or using fixed range
    // For TPC-H Q8, years are 1995-1996 from the date filter, but reference uses extract(year) so range is 1992-1998
    // We'll scan the first block to find range
    int32_t min_year = 1995, max_year = 1996; // Conservative for the date range
    // Actually let's scan the whole array with a quick kernel to find non-zero entries
    // Simplest: just check positions 0..9 and compact
    int32_t num_years = 0;
    auto num_years_dev = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    int compact_blocks = (max_years + 255) / 256;
    compact_kernel<<<compact_blocks, 256>>>(
        sum_brazil_arr.data_ptr<double>(),
        sum_total_arr.data_ptr<double>(),
        out_years.data_ptr<int32_t>(),
        out_mkt_share.data_ptr<double>(),
        num_years_dev.data_ptr<int32_t>(),
        1992, 1998
    );
    
    // Since we output all years, we need a proper compaction. For simplicity and correctness,
    // let's copy back and sort (only 6 elements, CPU is fine).
    // But output tensor must be filled correctly. Use a simple approach: 
    // Collect all non-zero years from host
    // Actually, let's just use the approach: output exactly the years found.
    // After compact kernel, we have unsorted output. The reference expects sorted years.
    // We'll sort on GPU after.
    
    // Since compact kernel atomically adds to num_years and writes, we can then sort
    // But for small data, just do it with a simple thrust call or built-in sort.
    // We'll just output as-is and let the Python sort wrapper handle it.
    // Actually, let's do a proper second kernel that sorts by year.
}

double q8_launcher_wrapper(
    torch::Tensor o_year,
    torch::Tensor volume,
    torch::Tensor is_brazil,
    torch::Tensor out_years,
    torch::Tensor out_mkt_share
) {
    q8_launcher(o_year, volume, is_brazil, out_years, out_mkt_share);
    return 0.0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_launcher_wrapper", &q8_launcher_wrapper, "TPC-H Q8 market share (CUDA)");
}
'''

_CUDA_SOURCE_FIXED = r'''
#include <torch/extension.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 1024

__global__ void q8_scatter_kernel(
    const int32_t* __restrict__ o_year_ptr,
    const double* __restrict__ volume_ptr,
    const bool* __restrict__ is_brazil_ptr,
    double* __restrict__ sum_brazil_out,
    double* __restrict__ sum_total_out,
    int32_t* __restrict__ year_flags_out,
    int64_t n_elements
) {
    // Each thread reads an element and atomically adds to per-year bins
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    int32_t yr = o_year_ptr[idx];
    double vol = volume_ptr[idx];
    double brazil_vol = is_brazil_ptr[idx] ? vol : 0.0;
    
    // Use year offset from 1990 (covers TPC-H range)
    int bin = yr - 1990;
    atomicAdd(&sum_brazil_out[bin], brazil_vol);
    atomicAdd(&sum_total_out[bin], vol);
    if (bin >= 0 && bin < 100) {
        atomicExch(&year_flags_out[bin], 1);
    }
}

__global__ void q8_compact_kernel(
    double* __restrict__ sum_brazil_arr,
    double* __restrict__ sum_total_arr,
    int32_t* __restrict__ year_flags,
    int32_t* __restrict__ out_years,
    double* __restrict__ out_mkt_share,
    int32_t* __restrict__ num_years_dev,
    int max_bins
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= max_bins) return;
    
    if (year_flags[tid] > 0) {
        double total = sum_total_arr[tid];
        if (total > 0.0) {
            int pos = atomicAdd(num_years_dev, 1);
            out_years[pos] = 1990 + tid;
            out_mkt_share[pos] = sum_brazil_arr[tid] / total;
        }
    }
}

void q8_launcher(
    torch::Tensor o_year,
    torch::Tensor volume,
    torch::Tensor is_brazil,
    torch::Tensor out_years,
    torch::Tensor out_mkt_share
) {
    TORCH_CHECK(o_year.scalar_type() == torch::kInt32);
    TORCH_CHECK(volume.scalar_type() == torch::kFloat64);
    TORCH_CHECK(is_brazil.scalar_type() == torch::kBool);
    TORCH_CHECK(out_years.scalar_type() == torch::kInt32);
    TORCH_CHECK(out_mkt_share.scalar_type() == torch::kFloat64);
    
    if (!o_year.is_cuda()) o_year = o_year.cuda();
    if (!volume.is_cuda()) volume = volume.cuda();
    if (!is_brazil.is_cuda()) is_brazil = is_brazil.cuda();
    
    o_year = o_year.contiguous();
    volume = volume.contiguous();
    is_brazil = is_brazil.contiguous();
    
    int64_t n = o_year.numel();
    int max_bins = 100; // covers years 1990-2089
    
    // Allocate temporary arrays
    auto sum_brazil_arr = torch::zeros({max_bins}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto sum_total_arr = torch::zeros({max_bins}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto year_flags = torch::zeros({max_bins}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    
    q8_scatter_kernel<<<blocks, threads>>>(
        o_year.data_ptr<int32_t>(),
        volume.data_ptr<double>(),
        is_brazil.data_ptr<bool>(),
        sum_brazil_arr.data_ptr<double>(),
        sum_total_arr.data_ptr<double>(),
        year_flags.data_ptr<int32_t>(),
        n
    );
    
    // Compact results
    auto num_years_dev = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    int compact_blocks = (max_bins + 255) / 256;
    q8_compact_kernel<<<compact_blocks, 256>>>(
        sum_brazil_arr.data_ptr<double>(),
        sum_total_arr.data_ptr<double>(),
        year_flags.data_ptr<int32_t>(),
        out_years.data_ptr<int32_t>(),
        out_mkt_share.data_ptr<double>(),
        num_years_dev.data_ptr<int32_t>(),
        max_bins
    );
    
    // Sort by year using thrust
    int num_years_host;
    cudaMemcpy(&num_years_host, num_years_dev.data_ptr<int32_t>(), sizeof(int), cudaMemcpyDeviceToHost);
    
    thrust::sort_by_key(
        thrust::device,
        out_years.data_ptr<int32_t>(),
        out_years.data_ptr<int32_t>() + num_years_host,
        out_mkt_share.data_ptr<double>()
    );
    
    // Zero out remaining entries
    if (num_years_host < out_years.numel()) {
        cudaMemset(out_years.data_ptr<int32_t>() + num_years_host, 0, (out_years.numel() - num_years_host) * sizeof(int32_t));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_launcher", &q8_launcher, "TPC-H Q8 market share (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q8_cuda_")
    cu_path = os.path.join(tmpdir, "_q8_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE_FIXED)
    return load(
        name="q8_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _launch_q8(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launches CUDA kernel for Q8 aggregation.
    
    Args:
        o_year: int32 tensor of years
        volume: float64 tensor of volume
        is_brazil: bool tensor indicating Brazil
    
    Returns:
        Tuple of (years, market_share) tensors sorted by year
    """
    n = o_year.shape[0]
    # Max unique years is small (<=5 for the date range), allocate safe buffer
    max_years = 20
    
    out_years = torch.zeros(max_years, dtype=torch.int32, device=o_year.device)
    out_mkt_share = torch.zeros(max_years, dtype=torch.float64, device=o_year.device)
    
    _EXT.q8_launcher(o_year, volume, is_brazil, out_years, out_mkt_share)
    
    # Find actual number of years used
    mask = out_years != 0
    valid_years = out_years[mask]
    valid_share = out_mkt_share[mask]
    
    if valid_years.shape[0] == 0:
        return torch.tensor([], dtype=torch.int32, device=o_year.device), torch.tensor([], dtype=torch.float64, device=o_year.device)
    
    # Sort by year (already sorted by thrust, but just in case)
    order = torch.argsort(valid_years)
    return valid_years[order], valid_share[order]


def _query_core(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core aggregation: compute market share per year.
    
    All tensors are 1D on GPU. This uses a fused CUDA kernel that:
    1. Scatters to per-year bins with atomicAdd
    2. Compacts non-zero bins
    3. Sorts by year using thrust
    
    This is faster than the reference because it avoids:
    - Multiple round-trips to cuDF/Python
    - Separate unique, scatter_add, and sort kernels
    - Temporary intermediate tensors
    
    Args:
        o_year: int32, shape (N,)
        volume: float64, shape (N,)
        is_brazil: bool, shape (N,)
    
    Returns:
        Tuple of (years, market_share) sorted by year
    """
    return _launch_q8(o_year, volume, is_brazil)


def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "AMERICA",
    part_type: str = "ECONOMY ANODIZED STEEL",
    nation_name: str = "BRAZIL",
    orderdate_lo: str = "1995-01-01",
    orderdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_year, sum(CASE WHEN nation='BRAZIL' THEN volume ELSE 0 END)/sum(volume) AS mkt_share
         FROM (SELECT extract(year FROM o_orderdate) AS o_year,
                      l_extendedprice*(1-l_discount) AS volume,
                      n2.n_name AS nation
               FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
               WHERE p_partkey=l_partkey AND s_suppkey=l_suppkey AND l_orderkey=o_orderkey
                 AND o_custkey=c_custkey AND c_nationkey=n1.n_nationkey
                 AND n1.n_regionkey=r_regionkey AND r_name='AMERICA'
                 AND s_nationkey=n2.n_nationkey
                 AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
                 AND p_type='ECONOMY ANODIZED STEEL') AS all_nations
         GROUP BY o_year ORDER BY o_year;
    Description: TPC-H Q8 — market share of a given nation within a region for a specific part type over two years.
    Params:
        region_name (str): region filter for customer's nation.
        part_type (str): part type filter.
        nation_name (str): nation whose market share is computed (supplier nation).
        orderdate_lo (str): order date lower bound (inclusive).
        orderdate_hi (str): order date upper bound (inclusive).
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    # Date bounds as epoch-day int32
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter region -> nation n1 (customer side)
    region_f = df_region[df_region["r_name"] == region_name][["r_regionkey"]]
    n1 = df_nation.merge(region_f, left_on="n_regionkey", right_on="r_regionkey")[
        ["n_nationkey"]
    ].rename(columns={"n_nationkey": "n1_nationkey"})

    # Customers in that region
    cust_r = df_customer.merge(n1, left_on="c_nationkey", right_on="n1_nationkey")[
        ["c_custkey"]
    ]

    # Filter orders by date range
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days <= hi_days)][
        ["o_orderkey", "o_custkey", "o_orderdate"]
    ]
    # Orders by customers in region
    orders_f = orders_f.merge(cust_r, left_on="o_custkey", right_on="c_custkey")[
        ["o_orderkey", "o_orderdate"]
    ]

    # Filter part by type
    part_f = df_part[df_part["p_type"] == part_type][["p_partkey"]]

    # Lineitem joined with part (filter by p_partkey)
    li = df_lineitem.merge(part_f, left_on="l_partkey", right_on="p_partkey")[
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with filtered orders
    li = li.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[
        ["l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"]
    ]

    # Join supplier -> nation n2 to get supplier nation name
    sup_n2 = df_supplier.merge(
        df_nation, left_on="s_nationkey", right_on="n_nationkey"
    )[["s_suppkey", "n_name"]]

    li = li.merge(sup_n2, left_on="l_suppkey", right_on="s_suppkey")[
        ["l_extendedprice", "l_discount", "o_orderdate", "n_name"]
    ]

    # Build tensors
    o_year_series = (li["o_orderdate"].dt.year).astype("int32")
    o_year_t = torch.as_tensor(o_year_series.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64)
    volume_t = price_t * (1.0 - disc_t)

    is_brazil_t = torch.as_tensor(
        (li["n_name"] == nation_name).astype("bool").to_cupy()
    )

    years_t, mkt_share_t = _query_core(o_year_t, volume_t, is_brazil_t)

    return cudf.DataFrame({
        "o_year": years_t,
        "mkt_share": mkt_share_t,
    })
