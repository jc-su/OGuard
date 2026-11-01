import os
import tempfile
from typing import Any

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 256
#define YEAR_BASE 1900
#define NUM_YEARS 201

__global__ void q8_groupby_kernel(
    const int32_t* __restrict__ o_year,
    const double* __restrict__ volume,
    const bool* __restrict__ is_brazil,
    double* __restrict__ sums,
    int64_t n_elements
) {
    extern __shared__ double shmem[];
    double* sh_total = shmem;
    double* sh_brazil = shmem + NUM_YEARS;

    int tid = threadIdx.x;

    for (int i = tid; i < NUM_YEARS; i += blockDim.x) {
        sh_total[i] = 0.0;
        sh_brazil[i] = 0.0;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    if (idx < n_elements) {
        int year_idx = static_cast<int>(o_year[idx]) - YEAR_BASE;
        if (year_idx >= 0 && year_idx < NUM_YEARS) {
            double v = volume[idx];
            atomicAdd(&sh_total[year_idx], v);
            if (is_brazil[idx]) {
                atomicAdd(&sh_brazil[year_idx], v);
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < NUM_YEARS; i += blockDim.x) {
        double t = sh_total[i];
        double b = sh_brazil[i];
        if (t != 0.0) {
            atomicAdd(&sums[i], t);
        }
        if (b != 0.0) {
            atomicAdd(&sums[NUM_YEARS + i], b);
        }
    }
}

std::vector<torch::Tensor> q8_groupby_launcher(
    torch::Tensor o_year,
    torch::Tensor volume,
    torch::Tensor is_brazil
) {
    TORCH_CHECK(o_year.scalar_type() == torch::kInt32, "o_year must be int32");
    TORCH_CHECK(volume.scalar_type() == torch::kFloat64, "volume must be float64");
    TORCH_CHECK(is_brazil.scalar_type() == torch::kBool, "is_brazil must be bool");
    TORCH_CHECK(o_year.numel() == volume.numel(), "o_year and volume size mismatch");
    TORCH_CHECK(o_year.numel() == is_brazil.numel(), "o_year and is_brazil size mismatch");

    if (!o_year.is_cuda()) {
        o_year = o_year.cuda();
    }
    if (!volume.is_cuda()) {
        volume = volume.cuda();
    }
    if (!is_brazil.is_cuda()) {
        is_brazil = is_brazil.cuda();
    }

    c10::cuda::CUDAGuard device_guard(volume.device());

    o_year = o_year.contiguous();
    volume = volume.contiguous();
    is_brazil = is_brazil.contiguous();

    int64_t n = volume.numel();

    auto sum_options = volume.options().dtype(torch::kFloat64);
    torch::Tensor sums = torch::zeros({2, NUM_YEARS}, sum_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n > 0) {
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        size_t shmem_bytes = static_cast<size_t>(2 * NUM_YEARS) * sizeof(double);

        q8_groupby_kernel<<<blocks, BLOCK_SIZE, shmem_bytes, stream>>>(
            o_year.data_ptr<int32_t>(),
            volume.data_ptr<double>(),
            is_brazil.data_ptr<bool>(),
            sums.data_ptr<double>(),
            n
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    std::vector<double> h_sums(2 * NUM_YEARS, 0.0);
    C10_CUDA_CHECK(cudaMemcpyAsync(
        h_sums.data(),
        sums.data_ptr<double>(),
        static_cast<size_t>(2 * NUM_YEARS) * sizeof(double),
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    int out_count = 0;
    for (int i = 0; i < NUM_YEARS; ++i) {
        if (h_sums[i] != 0.0) {
            ++out_count;
        }
    }

    auto year_options = o_year.options().dtype(torch::kInt32);
    auto share_options = volume.options().dtype(torch::kFloat64);
    torch::Tensor out_years = torch::empty({out_count}, year_options);
    torch::Tensor out_shares = torch::empty({out_count}, share_options);

    if (out_count > 0) {
        std::vector<int32_t> h_years(out_count);
        std::vector<double> h_shares(out_count);

        int pos = 0;
        for (int i = 0; i < NUM_YEARS; ++i) {
            double total = h_sums[i];
            if (total != 0.0) {
                double brazil = h_sums[NUM_YEARS + i];
                h_years[pos] = static_cast<int32_t>(YEAR_BASE + i);
                h_shares[pos] = brazil / total;
                ++pos;
            }
        }

        C10_CUDA_CHECK(cudaMemcpyAsync(
            out_years.data_ptr<int32_t>(),
            h_years.data(),
            static_cast<size_t>(out_count) * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream
        ));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            out_shares.data_ptr<double>(),
            h_shares.data(),
            static_cast<size_t>(out_count) * sizeof(double),
            cudaMemcpyHostToDevice,
            stream
        ));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return {out_years, out_shares};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_groupby_launcher", &q8_groupby_launcher, "TPC-H Q8 year market-share aggregation (CUDA)");
}
'''


def _load_cuda() -> Any:
    """
    Compile and load the CUDA extension from an embedded .cu source string.

    The device code performs the GPU hot path for `_query_core`: a coalesced scan of
    `o_year`, `volume`, and `is_brazil`, with per-block shared-memory accumulation
    and one global atomic flush per year per block. This replaces PyTorch
    `unique`/`scatter_add_` temporaries with a fused CUDA aggregation.
    """
    tmpdir = tempfile.mkdtemp(prefix="q8_cuda_")
    cu_path = os.path.join(tmpdir, "_q8_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q8_marketshare_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _launch_q8_groupby(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch the CUDA year-grouped aggregation and return sorted year/share tensors.

    Runs on GPU for the scan/reduction and uses only a tiny host-side finalization
    inside the C++ launcher to compact non-empty year groups. This is faster than
    the reference eager Torch path because it avoids `torch.unique`, inverse-index
    materialization, `where`, and two full scatter-add passes over the joined rows.
    """
    out = _EXT.q8_groupby_launcher(o_year, volume, is_brazil)
    return out[0], out[1]


def _query_core(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate `volume` by `o_year` and compute Brazil market share.

    GPU/CUDA hot path: a custom CUDA kernel scans the aligned 1D tensors once,
    accumulates total and Brazil volume per year in shared memory, then flushes
    block partials with global atomics. The output is sorted by year ascending,
    matching the reference semantics for the TPC-H date range while reducing
    memory traffic and intermediate tensor allocation versus eager PyTorch.
    """
    return _launch_q8_groupby(o_year, volume, is_brazil)


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
