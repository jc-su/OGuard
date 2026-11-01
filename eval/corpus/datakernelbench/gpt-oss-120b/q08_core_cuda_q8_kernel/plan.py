import os
import tempfile

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source – one fused kernel for Q8 aggregation
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define BLOCK_SIZE 1024

// ------------------------------------------------------------------
// q8_kernel
//   year         : int32_t per row
//   volume       : double per row
//   is_brazil    : uint8_t (0/1) per row
//   min_year     : int32_t, smallest year value
//   sum_total    : double[num_years] global accumulator
//   sum_brazil   : double[num_years] global accumulator
// ------------------------------------------------------------------
__global__ void q8_kernel(
    const int32_t* __restrict__ year,
    const double*  __restrict__ volume,
    const uint8_t* __restrict__ is_brazil,
    const int32_t min_year,
    double* __restrict__ sum_total,
    double* __restrict__ sum_brazil,
    const int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int32_t y = year[idx];
    int32_t bin = y - min_year;               // 0 .. num_years-1
    double vol = volume[idx];
    atomicAdd(&sum_total[bin], vol);
    if (is_brazil[idx])
        atomicAdd(&sum_brazil[bin], vol);
}

// ------------------------------------------------------------------
// Launcher
// ------------------------------------------------------------------
std::vector<torch::Tensor> q8_launcher(
    torch::Tensor year,
    torch::Tensor volume,
    torch::Tensor is_brazil,
    int32_t min_year,
    int32_t num_years)
{
    // sanity checks
    TORCH_CHECK(year.is_cuda(), "year must be CUDA tensor");
    TORCH_CHECK(volume.is_cuda(), "volume must be CUDA tensor");
    TORCH_CHECK(is_brazil.is_cuda(), "is_brazil must be CUDA tensor");
    TORCH_CHECK(year.scalar_type()   == torch::kInt32,   "year dtype must be int32");
    TORCH_CHECK(volume.scalar_type() == torch::kFloat64, "volume dtype must be float64");
    TORCH_CHECK(is_brazil.scalar_type() == torch::kUInt8, "is_brazil dtype must be uint8");

    // ensure contiguity
    year = year.contiguous();
    volume = volume.contiguous();
    is_brazil = is_brazil.contiguous();

    int64_t N = year.numel();

    // output buffers: one entry per distinct year
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor sum_total  = torch::zeros({num_years}, options);
    torch::Tensor sum_brazil = torch::zeros({num_years}, options);

    // launch configuration
    const int threads = BLOCK_SIZE;
    const int blocks  = (N + threads - 1) / threads;

    // kernel launch
    q8_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        year.data_ptr<int32_t>(),
        volume.data_ptr<double>(),
        is_brazil.data_ptr<uint8_t>(),
        min_year,
        sum_total.data_ptr<double>(),
        sum_brazil.data_ptr<double>(),
        N
    );
    // note: any CUDA errors will be propagated on the next sync point
    cudaDeviceSynchronize();

    return {sum_total, sum_brazil};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_launcher", &q8_launcher,
          "Q8 market‑share fused CUDA kernel (year → total, brazil)");
}
'''

# ----------------------------------------------------------------------
# Helper to compile the CUDA extension once per import
# ----------------------------------------------------------------------
def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q8_cuda_")
    cu_path = os.path.join(tmpdir, "q8_kernel.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q8_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        verbose=False,
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()   # compiled CUDA extension


# ----------------------------------------------------------------------
# Private launcher wrapper (Python side)
# ----------------------------------------------------------------------
def _launch_q8(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch the fused CUDA kernel that aggregates ``volume`` per distinct year
    and also per year for rows where ``is_brazil`` is true.

    Returns
    -------
    years : torch.Tensor[int32]   (sorted ascending)
    mkt_share : torch.Tensor[float64] (same order as ``years``)
    """
    # Determine the year range – cheap on‑device reductions
    min_year = int(o_year.min().item())
    max_year = int(o_year.max().item())
    num_years = max_year - min_year + 1

    # Cast bool to uint8 because the kernel expects uint8
    is_brazil_u8 = is_brazil.to(torch.uint8)

    # Call the compiled kernel
    sum_total, sum_brazil = _EXT.q8_launcher(
        o_year,
        volume,
        is_brazil_u8,
        min_year,
        num_years,
    )

    # Build the year index tensor
    years = torch.arange(
        min_year, max_year + 1, dtype=torch.int32, device=volume.device
    )
    # Compute market share; protect against division‑by‑zero
    mkt_share = torch.where(
        sum_total == 0,
        torch.zeros_like(sum_total),
        sum_brazil / sum_total,
    )
    return years, mkt_share


# ----------------------------------------------------------------------
# Public core API (unchanged signature)
# ----------------------------------------------------------------------
def _query_core(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core computation for TPCH Q8 – replaced with a single fused CUDA kernel.
    Input tensors must be 1‑D, on the CUDA device, and have dtypes:
      o_year    : int32
      volume    : float64
      is_brazil : bool (or any torch boolean dtype)
    Returns
    -------
    years : int32 tensor, sorted ascending
    mkt_share : float64 tensor, same order as ``years``
    """
    return _launch_q8(o_year, volume, is_brazil)


# ----------------------------------------------------------------------
# Public query entry point – unchanged from reference plan
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "AMERICA",
    part_type: str = "ECONOMY ANODIZED STEEL",
    nation_name: str = "BRAZIL",
    orderdate_lo: str = "1995-01-01",
    orderdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_year,
               sum(CASE WHEN nation='BRAZIL' THEN volume ELSE 0 END) / sum(volume)
          AS mkt_share
          FROM (
               SELECT extract(year FROM o_orderdate) AS o_year,
                      l_extendedprice*(1-l_discount) AS volume,
                      n2.n_name AS nation
               FROM part, supplier, lineitem, orders, customer,
                    nation n1, nation n2, region
               WHERE p_partkey=l_partkey
                 AND s_suppkey=l_suppkey
                 AND l_orderkey=o_orderkey
                 AND o_custkey=c_custkey
                 AND c_nationkey=n1.n_nationkey
                 AND n1.n_regionkey=r_regionkey
                 AND r_name='AMERICA'
                 AND s_nationkey=n2.n_nationkey
                 AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
                 AND p_type='ECONOMY ANODIZED STEEL') AS all_nations
         GROUP BY o_year ORDER BY o_year;
    Description: TPC‑H Q8 — market share of a given nation within a region for a
    specific part type over two years.
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

    # Date bounds as epoch‑day int32
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Region → nation (customer side)
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
    # Orders belonging to customers in the region
    orders_f = orders_f.merge(cust_r, left_on="o_custkey", right_on="c_custkey")[
        ["o_orderkey", "o_orderdate"]
    ]

    # Part filter
    part_f = df_part[df_part["p_type"] == part_type][["p_partkey"]]

    # Lineitem + part
    li = df_lineitem.merge(part_f, left_on="l_partkey", right_on="p_partkey")[
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
    ]

    # Join with filtered orders
    li = li.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[
        ["l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"]
    ]

    # Supplier → nation (supplier side)
    sup_n2 = df_supplier.merge(
        df_nation, left_on="s_nationkey", right_on="n_nationkey"
    )[["s_suppkey", "n_name"]]

    li = li.merge(sup_n2, left_on="l_suppkey", right_on="s_suppkey")[
        ["l_extendedprice", "l_discount", "o_orderdate", "n_name"]
    ]

    # Build tensors
    o_year_series = (li["o_orderdate"].dt.year).astype("int32")
    o_year_t = torch.as_tensor(o_year_series.to_cupy(), dtype=torch.int32, device="cuda")
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64, device="cuda")
    disc_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64, device="cuda")
    volume_t = price_t * (1.0 - disc_t)

    is_brazil_t = torch.as_tensor(
        (li["n_name"] == nation_name).astype("bool").to_cupy(),
        dtype=torch.bool,
        device="cuda",
    )

    years_t, mkt_share_t = _query_core(o_year_t, volume_t, is_brazil_t)

    return cudf.DataFrame({
        "o_year": years_t,
        "mkt_share": mkt_share_t,
    })
