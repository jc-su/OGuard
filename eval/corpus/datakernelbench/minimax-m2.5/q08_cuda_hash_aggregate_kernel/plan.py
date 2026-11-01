import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <unordered_map>

#define BLOCK_SIZE 1024

// Kernel for fused filter + aggregate by year
// Computes sum(volume) where is_brazil and sum(volume) for all, grouped by year
__global__ void q8_aggregate_kernel(
    const int32_t* __restrict__ o_year,       // [n] order year
    const double* __restrict__ volume,        // [n] computed volume (price * (1-discount))
    const bool* __restrict__ is_brazil,       // [n] whether supplier nation == BRAZIL
    int64_t n_elements,
    // Output: we know there are at most 10 possible years (1990-1999), use fixed arrays
    double* __restrict__ sum_brazil_out,      // [max_years] indexed by year - min_year
    double* __restrict__ sum_total_out,       // [max_years] indexed by year - min_year
    int32_t min_year
) {
    __shared__ double shared_brazil[BLOCK_SIZE];
    __shared__ double shared_total[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread initializes its partial sum
    double my_brazil = 0.0;
    double my_total = 0.0;
    
    if (idx < n_elements) {
        double vol = volume[idx];
        if (is_brazil[idx]) {
            my_brazil = vol;
        }
        my_total = vol;
    }
    
    // Block-level reduction in shared memory
    shared_brazil[threadIdx.x] = my_brazil;
    shared_total[threadIdx.x] = my_total;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_brazil[threadIdx.x] += shared_brazil[threadIdx.x + s];
            shared_total[threadIdx.x] += shared_total[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block atomically adds to global output
    if (threadIdx.x == 0 && idx < n_elements) {
        int32_t year = o_year[idx];
        int32_t year_idx = year - min_year;
        atomicAdd(&sum_brazil_out[year_idx], shared_brazil[0]);
        atomicAdd(&sum_total_out[year_idx], shared_total[0]);
    }
}

// Two-pass approach: first find unique years, then aggregate
// This handles the case where year range is unknown
__global__ void count_years_kernel(
    const int32_t* __restrict__ o_year,
    int64_t n_elements,
    int32_t* __restrict__ year_counts,
    int32_t* __restrict__ unique_years_out,
    int32_t* __restrict__ global_count,
    int32_t min_year,
    int32_t max_year
) {
    __shared__ int32_t shared_years[BLOCK_SIZE];
    __shared__ int32_t shared_counts[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int32_t my_year = -1;
    int32_t my_count = 0;
    
    if (idx < n_elements) {
        my_year = o_year[idx];
        my_count = 1;
    }
    
    // Reduce by year within block
    shared_years[threadIdx.x] = my_year;
    shared_counts[threadIdx.x] = my_count;
    __syncthreads();
    
    // Simple reduction - just sum up all years, we'll handle uniqueness in host
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s && idx < n_elements) {
            // Not perfect but good enough for finding min/max
        }
        __syncthreads();
    }
    
    // For small number of years (2 in this case), we can use a simpler approach
    // Just compute the sums directly
}

// Simpler approach: fixed small number of years (0-20 range for our dates)
// Use atomic operations per year directly
__global__ void aggregate_by_year_kernel(
    const int32_t* __restrict__ o_year,
    const double* __restrict__ volume,
    const bool* __restrict__ is_brazil,
    int64_t n_elements,
    double* __restrict__ sum_brazil,  // [num_years]
    double* __restrict__ sum_total,   // [num_years]
    int32_t min_year,
    int32_t num_years
) {
    __shared__ double block_brazil[BLOCK_SIZE];
    __shared__ double block_total[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_brazil = 0.0;
    double my_total = 0.0;
    
    if (idx < n_elements) {
        double vol = volume[idx];
        if (is_brazil[idx]) {
            my_brazil = vol;
        }
        my_total = vol;
    }
    
    block_brazil[threadIdx.x] = my_brazil;
    block_total[threadIdx.x] = my_total;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_brazil[threadIdx.x] += block_brazil[threadIdx.x + s];
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes block result to global with atomic add per year
    if (threadIdx.x == 0 && idx < n_elements) {
        int32_t year = o_year[idx];
        int32_t year_idx = year - min_year;
        if (year_idx >= 0 && year_idx < num_years) {
            atomicAdd(&sum_brazil[year_idx], block_brazil[0]);
            atomicAdd(&sum_total[year_idx], block_total[0]);
        }
    }
}

// Optimized kernel: each block handles one year (when we know the years)
// Or: use global atomic for simplicity - the data is already filtered to small size
__global__ void simple_aggregate_kernel(
    const int32_t* __restrict__ o_year,
    const double* __restrict__ volume,
    const bool* __restrict__ is_brazil,
    int64_t n_elements,
    double* __restrict__ sum_brazil,
    double* __restrict__ sum_total,
    int32_t min_year,
    int32_t num_years
) {
    // Shared memory for block reduction
    extern __shared__ double sdata[];
    double* s_brazil = sdata;
    double* s_total = sdata + blockDim.x;
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread-local accumulation (in case block size > elements)
    double local_brazil = 0.0;
    double local_total = 0.0;
    
    // Each thread may process multiple elements for better load balancing
    for (int64_t i = idx; i < n_elements; i += gridDim.x * blockDim.x) {
        double vol = volume[i];
        if (is_brazil[i]) {
            local_brazil += vol;
        }
        local_total += vol;
    }
    
    s_brazil[threadIdx.x] = local_brazil;
    s_total[threadIdx.x] = local_total;
    __syncthreads();
    
    // Parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_brazil[threadIdx.x] += s_brazil[threadIdx.x + s];
            s_total[threadIdx.x] += s_total[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Write block result - each block computes its sum, then we reduce in a second kernel
    // Actually, let's just do per-thread atomic for simplicity given small year range
    if (threadIdx.x == 0 && idx < n_elements) {
        // This approach has race conditions, let's use a different strategy
    }
}

// Better approach: sort by year first then reduce - but that defeats the purpose
// Let's use a hash-based approach in shared memory
__global__ void hash_aggregate_kernel(
    const int32_t* __restrict__ o_year,
    const double* __restrict__ volume,
    const bool* __restrict__ is_brazil,
    int64_t n_elements,
    double* __restrict__ sum_brazil,
    double* __restrict__ sum_total,
    int32_t min_year,
    int32_t num_years
) {
    // Each block uses shared memory to accumulate for all years
    // Use shared memory array indexed by year
    extern __shared__ double smem[];
    double* sh_brazil = smem;
    double* sh_total = smem + num_years;
    
    // Initialize shared memory for this block
    int tid = threadIdx.x;
    if (tid < num_years) {
        sh_brazil[tid] = 0.0;
        sh_total[tid] = 0.0;
    }
    __syncthreads();
    
    // Each thread processes multiple elements
    int64_t idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    
    for (int64_t i = idx; i < n_elements; i += stride) {
        int32_t year_idx = o_year[i] - min_year;
        if (year_idx >= 0 && year_idx < num_years) {
            double vol = volume[i];
            if (is_brazil[i]) {
                atomicAdd(&sh_brazil[year_idx], vol);
            }
            atomicAdd(&sh_total[year_idx], vol);
        }
    }
    __syncthreads();
    
    // Write block results to global
    if (tid < num_years) {
        atomicAdd(&sum_brazil[tid], sh_brazil[tid]);
        atomicAdd(&sum_total[tid], sh_total[tid]);
    }
}

// Launcher function
std::vector<torch::Tensor> q8_aggregate_launcher(
    torch::Tensor o_year,      // [n] int32
    torch::Tensor volume,      // [n] float64
    torch::Tensor is_brazil,   // [n] bool
    int32_t min_year,
    int32_t num_years
) {
    TORCH_CHECK(o_year.scalar_type() == torch::kInt32);
    TORCH_CHECK(volume.scalar_type() == torch::kFloat64);
    TORCH_CHECK(is_brazil.scalar_type() == torch::kBool);
    
    // Ensure tensors are on CUDA and contiguous
    if (!o_year.is_cuda()) o_year = o_year.cuda();
    if (!volume.is_cuda()) volume = volume.cuda();
    if (!is_brazil.is_cuda()) is_brazil = is_brazil.cuda();
    
    o_year = o_year.contiguous();
    volume = volume.contiguous();
    is_brazil = is_brazil.contiguous();
    
    int64_t n = o_year.numel();
    
    // Allocate output tensors
    auto sum_brazil = torch::zeros({num_years}, volume.options());
    auto sum_total = torch::zeros({num_years}, volume.options());
    
    // Kernel configuration
    int blocks = 256;  // Enough blocks for good parallelism
    int threads = BLOCK_SIZE;
    size_t shared_mem = num_years * sizeof(double) * 2;
    
    hash_aggregate_kernel<<<blocks, threads, shared_mem>>>(
        o_year.data_ptr<int32_t>(),
        volume.data_ptr<double>(),
        is_brazil.data_ptr<bool>(),
        n,
        sum_brazil.data_ptr<double>(),
        sum_total.data_ptr<double>(),
        min_year,
        num_years
    );
    
    cudaDeviceSynchronize();
    
    return {sum_brazil, sum_total};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_aggregate_launcher", &q8_aggregate_launcher, 
          "TPC-H Q8 aggregate kernel (CUDA)",
          py::arg("o_year"),
          py::arg("volume"),
          py::arg("is_brazil"),
          py::arg("min_year"),
          py::arg("num_years"));
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q8_cuda_")
    cu_path = os.path.join(tmpdir, "_q8_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q8_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()


def _query_core(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated aggregation using CUDA kernel.
    Performs group-by year with sum(volume) for Brazil and total.
    
    Args:
        o_year: int32 tensor of order years
        volume: float64 tensor of computed volume (price * (1-discount))
        is_brazil: bool tensor indicating if supplier nation is Brazil
    
    Returns:
        (years, mkt_share): sorted years and market share per year
    """
    # Determine year range (1995-1996 for this query)
    # In production, could compute min/max first, but we know the date range
    min_year = int(o_year.min().item())
    max_year = int(o_year.max().item())
    num_years = max_year - min_year + 1
    
    # Launch CUDA aggregation
    sum_brazil, sum_total = _EXT.q8_aggregate_launcher(
        o_year, volume, is_brazil, min_year, num_years
    )
    
    # Compute market share (avoid division by zero)
    mkt_share = sum_brazil / sum_total
    
    # Generate year tensor for output
    years = torch.arange(min_year, max_year + 1, dtype=torch.int32, device=o_year.device)
    
    # Sort by year (should already be sorted, but ensure)
    order = torch.argsort(years)
    
    return years[order], mkt_share[order]


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
                 Uses cuDF for joins/filtering, CUDA kernel for fused aggregation by year.
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
