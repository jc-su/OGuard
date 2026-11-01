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

__global__ void q8_aggregate_kernel(
    const int32_t* __restrict__ year_idx_ptr,
    const double* __restrict__ volume_ptr,
    const bool* __restrict__ is_brazil_ptr,
    double* __restrict__ sum_brazil_ptr,
    double* __restrict__ sum_total_ptr,
    int64_t n_elements,
    int64_t num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        int32_t year_idx = year_idx_ptr[idx];
        double vol = volume_ptr[idx];
        bool is_br = is_brazil_ptr[idx];
        
        if (year_idx >= 0 && year_idx < num_groups) {
            atomicAdd(&sum_total_ptr[year_idx], vol);
            if (is_br) {
                atomicAdd(&sum_brazil_ptr[year_idx], vol);
            }
        }
    }
}

std::vector<torch::Tensor> q8_launcher(
    torch::Tensor year_idx,
    torch::Tensor volume,
    torch::Tensor is_brazil,
    int64_t num_groups
) {
    TORCH_CHECK(year_idx.scalar_type() == torch::kInt32, "year_idx must be int32");
    TORCH_CHECK(volume.scalar_type() == torch::kFloat64, "volume must be float64");
    TORCH_CHECK(is_brazil.scalar_type() == torch::kBool, "is_brazil must be bool");
    
    if (!year_idx.is_cuda()) year_idx = year_idx.cuda();
    if (!volume.is_cuda()) volume = volume.cuda();
    if (!is_brazil.is_cuda()) is_brazil = is_brazil.cuda();
    
    year_idx = year_idx.contiguous();
    volume = volume.contiguous();
    is_brazil = is_brazil.contiguous();
    
    int64_t n = year_idx.numel();
    
    auto sum_brazil = torch::zeros({num_groups}, volume.options());
    auto sum_total = torch::zeros({num_groups}, volume.options());
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q8_aggregate_kernel<<<blocks, BLOCK_SIZE>>>(
        year_idx.data_ptr<int32_t>(),
        volume.data_ptr<double>(),
        is_brazil.data_ptr<bool>(),
        sum_brazil.data_ptr<double>(),
        sum_total.data_ptr<double>(),
        n,
        num_groups
    );
    
    cudaDeviceSynchronize();
    
    return {sum_brazil, sum_total};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_launcher", &q8_launcher, "TPC-H Q8 market share aggregation (CUDA)");
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

def _launch_q8(
    year_idx: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch CUDA kernel for grouped aggregation.
    
    Runs on GPU: fused filter+aggregate in one kernel pass.
    Faster than reference: avoids intermediate tensor creation (brazil_volume),
    uses direct atomic accumulation instead of scatter_add.
    """
    results = _EXT.q8_launcher(year_idx, volume, is_brazil, num_groups)
    return results[0], results[1]

def _query_core(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute market share by year using CUDA aggregation.
    
    GPU execution: CUDA kernel performs grouped sum (brazil volume, total volume)
    in one fused pass. PyTorch handles unique year discovery and final sorting.
    
    Why faster: 
    - Fused kernel avoids intermediate brazil_volume tensor allocation
    - Direct atomic accumulation vs scatter_add reduces memory traffic
    - Single kernel launch for aggregation vs multiple torch ops
    
    Args:
        o_year: Order year as int32 tensor (actual year values like 1995, 1996)
        volume: Line item volume (extendedprice * (1-discount)) as float64
        is_brazil: Boolean mask for Brazil supplier nation
    
    Returns:
        Tuple of (unique_years sorted ascending, market_share per year)
    """
    # Find unique years and create inverse mapping (small cardinality, fast in PyTorch)
    unique_years, inv = torch.unique(o_year, return_inverse=True)
    num_groups = unique_years.shape[0]
    
    # Cast inv to int32 for CUDA kernel (torch.unique returns int64 by default)
    inv = inv.to(torch.int32)
    
    # CUDA aggregation: compute sum_brazil and sum_total per year index
    sum_brazil, sum_total = _launch_q8(inv, volume, is_brazil, num_groups)
    
    # Compute market share ratio
    mkt_share = sum_brazil / sum_total
    
    # Order by year ascending
    order = torch.argsort(unique_years)
    return unique_years[order], mkt_share[order]

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
