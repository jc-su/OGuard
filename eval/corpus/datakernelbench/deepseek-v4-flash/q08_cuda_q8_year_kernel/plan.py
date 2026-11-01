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
#include <cstdint>

#define BLOCK_SIZE 1024
#define MAX_GROUPS 64  // More than enough for years (max ~100 years)

// Fused kernel: filter, project, group by year, aggregate with condition
__global__ void q8_kernel(
    const int32_t* __restrict__ o_year_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    const int32_t* __restrict__ is_brazil_ptr,
    double* __restrict__ sum_brazil_out,
    double* __restrict__ sum_total_out,
    int32_t* __restrict__ unique_years_out,
    int32_t* __restrict__ num_groups_out,
    int64_t n_elements
) {
    // Shared memory for block-level reductions per group
    // Layout: [block_size] for brazil, [block_size] for total
    extern __shared__ double shared[];
    double* block_brazil = shared;
    double* block_total = shared + blockDim.x;

    // Per-thread accumulators for groups found in this thread
    // We'll use a small array approach: each thread accumulates for all possible groups
    // But MAX_GROUPS is small (<= 100), so we can do shared memory per group
    
    // For simplicity with many groups, each thread computes partials
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        int32_t year = o_year_ptr[idx];
        double volume = price_ptr[idx] * (1.0 - disc_ptr[idx]);
        double brazil_vol = is_brazil_ptr[idx] ? volume : 0.0;
        
        // First pass: find unique years (cooperative)
        // Use block shared memory for unique years
        // For simplicity, use global atomics per group (only once per block after reduce)
        
        block_brazil[threadIdx.x] = brazil_vol;
        block_total[threadIdx.x] = volume;
    } else {
        block_brazil[threadIdx.x] = 0.0;
        block_total[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // Block-level reduction for all groups (each block handles one group)
    // We need to map years to blocks: each block handles a specific year
    
    // Alternative: use atomic per block for each group
    // Actually, let's do per-block group assignment
    int group_idx = blockIdx.x;
    if (group_idx * BLOCK_SIZE + threadIdx.x < n_elements) {
        // This block will accumulate for a specific year
        // Find the year for this block
        // For simplicity, first block finds years, then each block handles one year
        
        // Simplified: compute partial sum for this block's data
        // Then reduce and atomicAdd to global per year
    }
    
    // Simple approach: block reduce then atomic add for each year
    // First reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_brazil[threadIdx.x] += block_brazil[threadIdx.x + s];
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Now have accumulated for this block's data range
    // But we need per-year accumulation... This requires knowledge of which years are in this block
    // Better approach: separate kernel
    
    // Fallback: each block processes its own data range, but we need per-year sums
    // For this fixed-size problem (small number of years), do a two-pass approach
    
    if (threadIdx.x == 0) {
        // This block's accumulated values (all years mixed)
        // We need to compute per-year sums - this simple block reduce won't work
    }
}

// Simpler approach: second kernel to do per-year accumulation
__global__ void q8_year_kernel(
    const int32_t* __restrict__ o_year_ptr,
    const double* __restrict__ volume_ptr,
    const int32_t* __restrict__ is_brazil_ptr,
    double* __restrict__ sum_brazil_out,
    double* __restrict__ sum_total_out,
    int32_t* __restrict__ unique_years,
    int32_t num_groups,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double block_brazil[MAX_GROUPS];
    __shared__ double block_total[MAX_GROUPS];
    
    if (threadIdx.x < MAX_GROUPS) {
        block_brazil[threadIdx.x] = 0.0;
        block_total[threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    if (idx < n_elements) {
        int32_t year = o_year_ptr[idx];
        double vol = volume_ptr[idx];
        double brazil_vol = is_brazil_ptr[idx] ? vol : 0.0;
        
        // Find group index for this year
        for (int g = 0; g < num_groups; g++) {
            if (unique_years[g] == year) {
                atomicAdd(&block_brazil[g], brazil_vol);
                atomicAdd(&block_total[g], vol);
                break;
            }
        }
    }
    __syncthreads();
    
    // Write block results to global
    if (threadIdx.x < num_groups) {
        atomicAdd(&sum_brazil_out[threadIdx.x], block_brazil[threadIdx.x]);
        atomicAdd(&sum_total_out[threadIdx.x], block_total[threadIdx.x]);
    }
}

// First kernel: compute volume and pass to year kernel
__global__ void q8_volume_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    double* __restrict__ volume_out,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        volume_out[idx] = price_ptr[idx] * (1.0 - disc_ptr[idx]);
    }
}

// Launcher that does everything
void q8_launcher(
    torch::Tensor o_year,
    torch::Tensor price,
    torch::Tensor disc,
    torch::Tensor is_brazil,
    torch::Tensor sum_brazil_out,
    torch::Tensor sum_total_out,
    torch::Tensor unique_years_out,
    int32_t num_groups,
    int64_t n_elements
) {
    // Make contiguous and ensure CUDA
    o_year = o_year.contiguous().cuda();
    price = price.contiguous().cuda();
    disc = disc.contiguous().cuda();
    is_brazil = is_brazil.contiguous().cuda();
    sum_brazil_out = sum_brazil_out.contiguous().cuda();
    sum_total_out = sum_total_out.contiguous().cuda();
    unique_years_out = unique_years_out.contiguous().cuda();
    
    // Compute volume
    auto volume = torch::empty({n_elements}, price.options());
    int blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q8_volume_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        disc.data_ptr<double>(),
        volume.data_ptr<double>(),
        n_elements
    );
    
    // Aggregate per year
    int year_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q8_year_kernel<<<year_blocks, BLOCK_SIZE, MAX_GROUPS * 2 * sizeof(double)>>>(
        o_year.data_ptr<int32_t>(),
        volume.data_ptr<double>(),
        is_brazil.data_ptr<int32_t>(),
        sum_brazil_out.data_ptr<double>(),
        sum_total_out.data_ptr<double>(),
        unique_years_out.data_ptr<int32_t>(),
        num_groups,
        n_elements
    );
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
    Uses cuDF for joins and string filters, then CUDA for fused volume computation and per-year aggregation.
    
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

    # Build tensors for CUDA
    o_year_series = (li["o_orderdate"].dt.year).astype("int32")
    o_year_t = torch.as_tensor(o_year_series.to_cupy(), dtype=torch.int32).contiguous()
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64).contiguous()
    disc_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64).contiguous()
    
    is_brazil_t = torch.as_tensor(
        (li["n_name"] == nation_name).astype("int32").to_cupy()
    ).contiguous()

    n_elements = o_year_t.shape[0]
    
    # Find unique years using PyTorch
    unique_years = torch.unique(o_year_t).contiguous()
    num_groups = unique_years.shape[0]
    
    # Initialize output tensors
    sum_brazil = torch.zeros(num_groups, dtype=torch.float64, device='cuda')
    sum_total = torch.zeros(num_groups, dtype=torch.float64, device='cuda')
    
    # Launch CUDA kernel
    _EXT.q8_launcher(
        o_year_t.cuda(),
        price_t.cuda(),
        disc_t.cuda(),
        is_brazil_t.cuda(),
        sum_brazil,
        sum_total,
        unique_years.cuda(),
        num_groups,
        n_elements
    )
    
    # Compute market share and sort
    mkt_share = sum_brazil / sum_total
    order = torch.argsort(unique_years)
    
    years_result = unique_years[order]
    mkt_share_result = mkt_share[order]
    
    return cudf.DataFrame({
        "o_year": years_result.cpu(),
        "mkt_share": mkt_share_result.cpu(),
    })
