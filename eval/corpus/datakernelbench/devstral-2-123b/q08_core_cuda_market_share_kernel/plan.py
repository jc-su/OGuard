import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void market_share_kernel(
    const int32_t* __restrict__ o_year,
    const double* __restrict__ volume,
    const bool* __restrict__ is_brazil,
    double* __restrict__ sum_brazil,
    double* __restrict__ sum_total,
    int64_t n_elements
) {
    __shared__ double shared_brazil[BLOCK_SIZE];
    __shared__ double shared_total[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_brazil = 0.0;
    double local_total = 0.0;

    if (idx < n_elements) {
        local_total = volume[idx];
        if (is_brazil[idx]) {
            local_brazil = local_total;
        }
    }

    shared_brazil[threadIdx.x] = local_brazil;
    shared_total[threadIdx.x] = local_total;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_brazil[threadIdx.x] += shared_brazil[threadIdx.x + stride];
            shared_total[threadIdx.x] += shared_total[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&sum_brazil[blockIdx.x], shared_brazil[0]);
        atomicAdd(&sum_total[blockIdx.x], shared_total[0]);
    }
}

void market_share_launcher(
    torch::Tensor o_year,
    torch::Tensor volume,
    torch::Tensor is_brazil,
    torch::Tensor sum_brazil,
    torch::Tensor sum_total
) {
    const int64_t n = o_year.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    market_share_kernel<<<blocks, BLOCK_SIZE>>>(
        o_year.data_ptr<int32_t>(),
        volume.data_ptr<double>(),
        is_brazil.data_ptr<bool>(),
        sum_brazil.data_ptr<double>(),
        sum_total.data_ptr<double>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("market_share_launcher", &market_share_launcher, "Q8 market share launcher");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q8_cuda_")
    cu_path = os.path.join(tmpdir, "_q8_kernel.cu")
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
    n = o_year.numel()
    blocks = (n + 1024 - 1) // 1024
    sum_brazil = torch.zeros(blocks, dtype=torch.float64, device=volume.device)
    sum_total = torch.zeros(blocks, dtype=torch.float64, device=volume.device)

    _EXT.market_share_launcher(
        o_year.contiguous(),
        volume.contiguous(),
        is_brazil.contiguous(),
        sum_brazil,
        sum_total
    )

    total_brazil = sum_brazil.sum()
    total_all = sum_total.sum()
    mkt_share = total_brazil / total_all

    unique_years, inv = torch.unique(o_year, return_inverse=True)
    sum_brazil = torch.zeros_like(unique_years, dtype=torch.float64)
    sum_total = torch.zeros_like(unique_years, dtype=torch.float64)
    sum_brazil.scatter_add_(0, inv, torch.where(is_brazil, volume, torch.zeros_like(volume)))
    sum_total.scatter_add_(0, inv, volume)
    mkt_share_per_year = sum_brazil / sum_total

    order = torch.argsort(unique_years)
    return unique_years[order], mkt_share_per_year[order]

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

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    region_f = df_region[df_region["r_name"] == region_name][["r_regionkey"]]
    n1 = df_nation.merge(region_f, left_on="n_regionkey", right_on="r_regionkey")[["n_nationkey"]].rename(columns={"n_nationkey": "n1_nationkey"})
    cust_r = df_customer.merge(n1, left_on="c_nationkey", right_on="n1_nationkey")[["c_custkey"]]
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days <= hi_days)][["o_orderkey", "o_custkey", "o_orderdate"]]
    orders_f = orders_f.merge(cust_r, left_on="o_custkey", right_on="c_custkey")[["o_orderkey", "o_orderdate"]]
    part_f = df_part[df_part["p_type"] == part_type][["p_partkey"]]
    li = df_lineitem.merge(part_f, left_on="l_partkey", right_on="p_partkey")[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]]
    li = li.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[["l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"]]
    sup_n2 = df_supplier.merge(df_nation, left_on="s_nationkey", right_on="n_nationkey")[["s_suppkey", "n_name"]]
    li = li.merge(sup_n2, left_on="l_suppkey", right_on="s_suppkey")[["l_extendedprice", "l_discount", "o_orderdate", "n_name"]]
    o_year_series = (li["o_orderdate"].dt.year).astype("int32")
    o_year_t = torch.as_tensor(o_year_series.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64)
    volume_t = price_t * (1.0 - disc_t)
    is_brazil_t = torch.as_tensor((li["n_name"] == nation_name).values, dtype=torch.bool)

    years_t, mkt_share_t = _query_core(o_year_t, volume_t, is_brazil_t)
    return cudf.DataFrame({"o_year": years_t.int(), "mkt_share": mkt_share_t})
