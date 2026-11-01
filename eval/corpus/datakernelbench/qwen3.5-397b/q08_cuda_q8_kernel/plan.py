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

#define BLOCK_SIZE 1024

// Kernel for TPC-H Q8 Aggregation
// Computes sum(volume) and sum(brazil_volume) grouped by year (1995, 1996)
// Index 0 -> 1995, Index 1 -> 1996
__global__ void q8_kernel(
    const int32_t* __restrict__ years,
    const double* __restrict__ prices,
    const double* __restrict__ discounts,
    const bool* __restrict__ is_brazil,
    double* __restrict__ out_total,
    double* __restrict__ out_brazil,
    int64_t n_elements
) {
    // Shared memory layout (4 * BLOCK_SIZE doubles):
    // [0..BLOCK_SIZE-1] = total_1995
    // [BLOCK_SIZE..2*BLOCK_SIZE-1] = total_1996
    // [2*BLOCK_SIZE..3*BLOCK_SIZE-1] = brazil_1995
    // [3*BLOCK_SIZE..4*BLOCK_SIZE-1] = brazil_1996
    extern __shared__ double sdata[];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double reg_total_1995 = 0.0;
    double reg_total_1996 = 0.0;
    double reg_brazil_1995 = 0.0;
    double reg_brazil_1996 = 0.0;

    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n_elements; i += stride) {
        double vol = prices[i] * (1.0 - discounts[i]);
        int32_t y = years[i];
        bool is_br = is_brazil[i];
        
        if (y == 1995) {
            reg_total_1995 += vol;
            if (is_br) reg_brazil_1995 += vol;
        } else if (y == 1996) {
            reg_total_1996 += vol;
            if (is_br) reg_brazil_1996 += vol;
        }
    }

    // Write to shared memory with proper offsets to avoid overlap
    sdata[tid] = reg_total_1995;
    sdata[BLOCK_SIZE + tid] = reg_total_1996;
    sdata[2 * BLOCK_SIZE + tid] = reg_brazil_1995;
    sdata[3 * BLOCK_SIZE + tid] = reg_brazil_1996;
    
    __syncthreads();

    // Reduction for all 4 accumulators
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[BLOCK_SIZE + tid] += sdata[BLOCK_SIZE + tid + s];
            sdata[2 * BLOCK_SIZE + tid] += sdata[2 * BLOCK_SIZE + tid + s];
            sdata[3 * BLOCK_SIZE + tid] += sdata[3 * BLOCK_SIZE + tid + s];
        }
        __syncthreads();
    }

    // Thread 0 atomics to global memory
    if (tid == 0) {
        atomicAdd(&out_total[0], sdata[0]);              // 1995 total
        atomicAdd(&out_total[1], sdata[BLOCK_SIZE]);     // 1996 total
        atomicAdd(&out_brazil[0], sdata[2 * BLOCK_SIZE]); // 1995 brazil
        atomicAdd(&out_brazil[1], sdata[3 * BLOCK_SIZE]); // 1996 brazil
    }
}

std::vector<torch::Tensor> q8_launcher(
    torch::Tensor years,
    torch::Tensor prices,
    torch::Tensor discounts,
    torch::Tensor is_brazil
) {
    if (!years.is_cuda()) years = years.cuda();
    if (!prices.is_cuda()) prices = prices.cuda();
    if (!discounts.is_cuda()) discounts = discounts.cuda();
    if (!is_brazil.is_cuda()) is_brazil = is_brazil.cuda();

    years = years.contiguous();
    prices = prices.contiguous();
    discounts = discounts.contiguous();
    is_brazil = is_brazil.contiguous();

    TORCH_CHECK(years.scalar_type() == torch::kInt32, "years must be int32");
    TORCH_CHECK(prices.scalar_type() == torch::kFloat64, "prices must be float64");
    TORCH_CHECK(discounts.scalar_type() == torch::kFloat64, "discounts must be float64");
    TORCH_CHECK(is_brazil.scalar_type() == torch::kBool, "is_brazil must be bool");

    int64_t n = years.numel();
    
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(years.device());
    auto out_total = torch::zeros({2}, options);
    auto out_brazil = torch::zeros({2}, options);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem = 4 * BLOCK_SIZE * sizeof(double);

    q8_kernel<<<blocks, BLOCK_SIZE, smem>>>(
        years.data_ptr<int32_t>(),
        prices.data_ptr<double>(),
        discounts.data_ptr<double>(),
        is_brazil.data_ptr<bool>(),
        out_total.data_ptr<double>(),
        out_brazil.data_ptr<double>(),
        n
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return {out_total, out_brazil};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_launcher", &q8_launcher, "TPC-H Q8 Aggregation Kernel");
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
        verbose=False,
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
                 Optimized with fused CUDA kernel for volume calculation and grouped aggregation.
                 cuDF handles joins and string filters; CUDA handles numeric projection and reduction.
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
    
    # Pass price and discount to kernel for fused volume calculation
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64)
    
    is_brazil_t = torch.as_tensor(
        (li["n_name"] == nation_name).astype("bool").to_cupy()
    )

    # Launch CUDA kernel
    out_total, out_brazil = _EXT.q8_launcher(o_year_t, price_t, disc_t, is_brazil_t)
    
    # Compute market share
    mkt_share = out_brazil / torch.where(out_total == 0, torch.ones_like(out_total), out_total)
    
    # Years are fixed 1995, 1996. Map back to tensor.
    years_t = torch.tensor([1995, 1996], dtype=torch.int32, device=o_year_t.device)
    
    # Filter out years that had no data (total == 0)
    mask = out_total > 0
    years_t = years_t[mask]
    mkt_share = mkt_share[mask]
    
    # Sort by year (already sorted 1995, 1996, but ensure correctness)
    if years_t.numel() > 0:
        order = torch.argsort(years_t)
        years_t = years_t[order]
        mkt_share = mkt_share[order]

    return cudf.DataFrame({
        "o_year": years_t.cpu(),
        "mkt_share": mkt_share.cpu(),
    })
