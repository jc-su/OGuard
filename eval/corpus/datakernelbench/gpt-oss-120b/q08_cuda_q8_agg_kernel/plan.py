import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused filter‑project‑aggregate for TPC‑H Q8
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ------------------------------------------------------------------
// Kernel: per‑year aggregation (sum total & sum for Brazil)
// ------------------------------------------------------------------
__global__ void q8_agg_kernel(
    const int32_t* __restrict__ years,
    const double* __restrict__ volume,
    const uint8_t* __restrict__ is_brazil,
    double* __restrict__ sum_total,
    double* __restrict__ sum_brazil,
    const int32_t year_lo,
    const int32_t num_years,
    const int64_t N)
{
    // shared memory holds two double arrays (one per aggregation)
    extern __shared__ double shmem[];
    double* sh_total  = shmem;                     // size = num_years
    double* sh_brazil = shmem + num_years;         // size = num_years

    // initialise shared buffers
    for (int i = threadIdx.x; i < num_years; i += blockDim.x) {
        sh_total[i]  = 0.0;
        sh_brazil[i] = 0.0;
    }
    __syncthreads();

    // grid‑stride loop over rows
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        int32_t y = years[i];
        int off = y - year_lo;                     // map to 0 … num_years‑1
        if (off >= 0 && off < num_years) {
            double vol = volume[i];
            atomicAdd(&sh_total[off], vol);
            if (is_brazil[i])
                atomicAdd(&sh_brazil[off], vol);
        }
    }
    __syncthreads();

    // flush per‑block partials to global output (one atomic per year)
    for (int i = threadIdx.x; i < num_years; i += blockDim.x) {
        if (sh_total[i] != 0.0)
            atomicAdd(&sum_total[i], sh_total[i]);
        if (sh_brazil[i] != 0.0)
            atomicAdd(&sum_brazil[i], sh_brazil[i]);
    }
}

// ------------------------------------------------------------------
// Launcher – prepares tensors, launches kernel, returns results
// ------------------------------------------------------------------
py::tuple q8_launcher(
    torch::Tensor years,
    torch::Tensor volume,
    torch::Tensor is_brazil,
    const int32_t year_lo,
    const int32_t num_years)
{
    TORCH_CHECK(years.is_cuda(), "years must be a CUDA tensor");
    TORCH_CHECK(volume.is_cuda(), "volume must be a CUDA tensor");
    TORCH_CHECK(is_brazil.is_cuda(), "is_brazil must be a CUDA tensor");
    TORCH_CHECK(years.dtype()   == torch::kInt32,   "years must be int32");
    TORCH_CHECK(volume.dtype()  == torch::kFloat64, "volume must be float64");
    TORCH_CHECK(is_brazil.dtype() == torch::kByte || is_brazil.dtype() == torch::kUInt8,
                "is_brazil must be uint8 (torch.kByte)");

    years = years.contiguous();
    volume = volume.contiguous();
    is_brazil = is_brazil.contiguous();

    const int64_t N = years.numel();
    auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(years.device());

    // output buffers: one element per distinct year in the range
    torch::Tensor sum_total  = torch::zeros({num_years}, opts);
    torch::Tensor sum_brazil = torch::zeros({num_years}, opts);

    const int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t shared_bytes = sizeof(double) * 2 * num_years;

    q8_agg_kernel<<<blocks, BLOCK_SIZE, shared_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        years.data_ptr<int32_t>(),
        volume.data_ptr<double>(),
        is_brazil.data_ptr<uint8_t>(),
        sum_total.data_ptr<double>(),
        sum_brazil.data_ptr<double>(),
        year_lo,
        num_years,
        N
    );

    // propagate any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        AT_ERROR(cudaGetErrorString(err));

    return py::make_tuple(sum_total, sum_brazil);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_launcher", &q8_launcher, "TPC-H Q8 fused aggregation (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Compile and load the CUDA extension (once per import)
# ----------------------------------------------------------------------
def _load_cuda_extension() -> any:
    """Compile the embedded CUDA source and return the loaded module."""
    tmpdir = tempfile.mkdtemp(prefix="q8_cuda_")
    cu_path = os.path.join(tmpdir, "q8_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)

    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    # H100 = sm_90
    ext = load(
        name="q8_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )
    return ext

_EXT = _load_cuda_extension()   # compile at import time

# ----------------------------------------------------------------------
# Core aggregation – thin wrapper around the CUDA launcher
# ----------------------------------------------------------------------
def _query_core(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
    year_lo: int,
    year_hi: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse filter‑project‑aggregate for Q8.

    Parameters
    ----------
    o_year : torch.Tensor[int32]
        Extracted order year (already filtered).
    volume : torch.Tensor[float64]
        ``l_extendedprice * (1 - l_discount)`` per row.
    is_brazil : torch.Tensor[uint8]
        Flag where supplier nation == ``nation_name``.
    year_lo / year_hi : int
        Inclusive bounds of the distinct years present.

    Returns
    -------
    tuple (years, mkt_share)
        *years* – ``torch.int32`` tensor sorted ascending.
        *mkt_share* – market‑share (double) per year.
    """
    num_years = year_hi - year_lo + 1
    sum_total, sum_brazil = _EXT.q8_launcher(
        o_year, volume, is_brazil, year_lo, num_years
    )
    years = torch.arange(year_lo, year_hi + 1, dtype=torch.int32, device=volume.device)
    mkt_share = sum_brazil / sum_total
    return years, mkt_share

# ----------------------------------------------------------------------
# Public API – unchanged signature
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
    SQL:
        SELECT o_year,
               sum(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) /
               sum(volume) AS mkt_share
        FROM (
            SELECT extract(year FROM o_orderdate) AS o_year,
                   l_extendedprice * (1 - l_discount) AS volume,
                   n2.n_name AS nation
            FROM part, supplier, lineitem, orders, customer,
                 nation n1, nation n2, region
            WHERE p_partkey = l_partkey
              AND s_suppkey = l_suppkey
              AND l_orderkey = o_orderkey
              AND o_custkey = c_custkey
              AND c_nationkey = n1.n_nationkey
              AND n1.n_regionkey = r_regionkey
              AND r_name = 'AMERICA'
              AND s_nationkey = n2.n_nationkey
              AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
              AND p_type = 'ECONOMY ANODIZED STEEL'
        ) AS all_nations
        GROUP BY o_year
        ORDER BY o_year;

    Description:
        cuDF executes all relational joins and filters.  The three
        columns (year, volume, is_brazil) are materialised as GPU
        tensors.  `_query_core` launches a single fused CUDA kernel that
        scans these columns once, aggregates per distinct year, and
        computes the market‑share.  This removes multiple cuDF‑to‑torch
        round‑trips and reduces HBM traffic—the dominant bottleneck
        for this memory‑bound scan on an H100.

    Params
    ------
    region_name : str
        Region filter for the customer side (default 'AMERICA').
    part_type : str
        Part type filter (default 'ECONOMY ANODIZED STEEL').
    nation_name : str
        Supplier nation whose market share is computed (default 'BRAZIL').
    orderdate_lo / orderdate_hi : str
        Inclusive lower/upper bounds for order dates (YYYY‑MM‑DD).
    """
    # ------------------------------------------------------------------
    # Load tables
    # ------------------------------------------------------------------
    df_part      = tables["part"]
    df_supplier  = tables["supplier"]
    df_lineitem  = tables["lineitem"]
    df_orders    = tables["orders"]
    df_customer  = tables["customer"]
    df_nation    = tables["nation"]
    df_region    = tables["region"]

    # ------------------------------------------------------------------
    # Date bounds as epoch‑day int32
    # ------------------------------------------------------------------
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # --------------------------------------------------------------
    # 1. Region → nation (customer side)   n1
    # --------------------------------------------------------------
    region_f = df_region[df_region["r_name"] == region_name][["r_regionkey"]]
    n1 = (
        df_nation.merge(region_f, left_on="n_regionkey", right_on="r_regionkey")
        [["n_nationkey"]]
        .rename(columns={"n_nationkey": "n1_nationkey"})
    )

    # --------------------------------------------------------------
    # 2. Customers in that region
    # --------------------------------------------------------------
    cust_r = df_customer.merge(n1, left_on="c_nationkey", right_on="n1_nationkey")[
        ["c_custkey"]
    ]

    # --------------------------------------------------------------
    # 3. Orders in date range and belonging to customers in region
    # --------------------------------------------------------------
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days <= hi_days)][
        ["o_orderkey", "o_custkey", "o_orderdate"]
    ]
    orders_f = orders_f.merge(cust_r, left_on="o_custkey", right_on="c_custkey")[
        ["o_orderkey", "o_orderdate"]
    ]

    # --------------------------------------------------------------
    # 4. Part filter
    # --------------------------------------------------------------
    part_f = df_part[df_part["p_type"] == part_type][["p_partkey"]]

    # --------------------------------------------------------------
    # 5. Lineitem ↔ filtered part
    # --------------------------------------------------------------
    li = df_lineitem.merge(part_f, left_on="l_partkey", right_on="p_partkey")[
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
    ]

    # --------------------------------------------------------------
    # 6. Join lineitem with filtered orders
    # --------------------------------------------------------------
    li = li.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[
        ["l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"]
    ]

    # --------------------------------------------------------------
    # 7. Supplier ↔ nation (supplier side) to obtain nation name
    # --------------------------------------------------------------
    sup_n2 = df_supplier.merge(df_nation, left_on="s_nationkey", right_on="n_nationkey")[
        ["s_suppkey", "n_name"]
    ]

    li = li.merge(sup_n2, left_on="l_suppkey", right_on="s_suppkey")[
        ["l_extendedprice", "l_discount", "o_orderdate", "n_name"]
    ]

    # --------------------------------------------------------------
    # 8. Build GPU tensors for the hot path
    # --------------------------------------------------------------
    # Extract year as int32
    o_year_series = (li["o_orderdate"].dt.year).astype("int32")
    o_year_t = torch.as_tensor(o_year_series.to_cupy(), dtype=torch.int32)

    # Compute volume = l_extendedprice * (1 - l_discount)
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t  = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64)
    volume_t = price_t * (1.0 - disc_t)

    # Flag for Brazil (supplier nation) – cast to uint8 (torch.kByte)
    is_brazil_t = torch.as_tensor(
        (li["n_name"] == nation_name).astype("uint8").to_cupy(),
        dtype=torch.uint8,
    )

    # Determine inclusive year bounds from the supplied dates
    year_lo = pd.to_datetime(orderdate_lo).year
    year_hi = pd.to_datetime(orderdate_hi).year

    # --------------------------------------------------------------
    # 9. Execute fused GPU aggregation
    # --------------------------------------------------------------
    years_t, mkt_share_t = _query_core(o_year_t, volume_t, is_brazil_t, year_lo, year_hi)

    # --------------------------------------------------------------
    # 10. Return result as cuDF DataFrame (column order matches reference)
    # --------------------------------------------------------------
    return cudf.DataFrame({
        "o_year": years_t,
        "mkt_share": mkt_share_t,
    })
