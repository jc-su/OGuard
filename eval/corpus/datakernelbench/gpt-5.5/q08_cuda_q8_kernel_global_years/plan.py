import os
import tempfile
from typing import Any

import cudf
import pandas as pd
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 1024
#define MAX_YEARS 128

__global__ void q8_kernel_two_year(
    const int32_t* __restrict__ year_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ suppkey_ptr,
    const uint8_t* __restrict__ target_supp_ptr,
    int64_t target_supp_len,
    int32_t base_year,
    double* __restrict__ out_target_ptr,
    double* __restrict__ out_total_ptr,
    int64_t* __restrict__ out_count_ptr,
    int64_t n_elements
) {
    __shared__ double s_total0[BLOCK_SIZE];
    __shared__ double s_total1[BLOCK_SIZE];
    __shared__ double s_target0[BLOCK_SIZE];
    __shared__ double s_target1[BLOCK_SIZE];
    __shared__ unsigned long long s_count0[BLOCK_SIZE];
    __shared__ unsigned long long s_count1[BLOCK_SIZE];

    int tid = threadIdx.x;
    double total0 = 0.0;
    double total1 = 0.0;
    double target0 = 0.0;
    double target1 = 0.0;
    unsigned long long count0 = 0ULL;
    unsigned long long count1 = 0ULL;

    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n_elements; idx += stride) {
        int32_t y = year_ptr[idx];
        int32_t g = y - base_year;
        if (g == 0 || g == 1) {
            double vol = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            int64_t sk = suppkey_ptr[idx];
            bool is_target = (sk >= 0 && sk < target_supp_len && target_supp_ptr[sk] != 0);
            if (g == 0) {
                total0 += vol;
                target0 += is_target ? vol : 0.0;
                count0 += 1ULL;
            } else {
                total1 += vol;
                target1 += is_target ? vol : 0.0;
                count1 += 1ULL;
            }
        }
    }

    s_total0[tid] = total0;
    s_total1[tid] = total1;
    s_target0[tid] = target0;
    s_target1[tid] = target1;
    s_count0[tid] = count0;
    s_count1[tid] = count1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_total0[tid] += s_total0[tid + s];
            s_total1[tid] += s_total1[tid + s];
            s_target0[tid] += s_target0[tid + s];
            s_target1[tid] += s_target1[tid + s];
            s_count0[tid] += s_count0[tid + s];
            s_count1[tid] += s_count1[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&out_total_ptr[0], s_total0[0]);
        atomicAdd(&out_total_ptr[1], s_total1[0]);
        atomicAdd(&out_target_ptr[0], s_target0[0]);
        atomicAdd(&out_target_ptr[1], s_target1[0]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&out_count_ptr[0]), s_count0[0]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&out_count_ptr[1]), s_count1[0]);
    }
}

__global__ void q8_kernel_shared_years(
    const int32_t* __restrict__ year_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ suppkey_ptr,
    const uint8_t* __restrict__ target_supp_ptr,
    int64_t target_supp_len,
    int32_t base_year,
    int32_t n_years,
    double* __restrict__ out_target_ptr,
    double* __restrict__ out_total_ptr,
    int64_t* __restrict__ out_count_ptr,
    int64_t n_elements
) {
    __shared__ double s_total[MAX_YEARS];
    __shared__ double s_target[MAX_YEARS];
    __shared__ unsigned long long s_count[MAX_YEARS];

    int tid = threadIdx.x;
    for (int g = tid; g < n_years; g += blockDim.x) {
        s_total[g] = 0.0;
        s_target[g] = 0.0;
        s_count[g] = 0ULL;
    }
    __syncthreads();

    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n_elements; idx += stride) {
        int32_t g = year_ptr[idx] - base_year;
        if (g >= 0 && g < n_years) {
            double vol = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            int64_t sk = suppkey_ptr[idx];
            bool is_target = (sk >= 0 && sk < target_supp_len && target_supp_ptr[sk] != 0);
            atomicAdd(&s_total[g], vol);
            if (is_target) {
                atomicAdd(&s_target[g], vol);
            }
            atomicAdd(&s_count[g], 1ULL);
        }
    }
    __syncthreads();

    for (int g = tid; g < n_years; g += blockDim.x) {
        atomicAdd(&out_total_ptr[g], s_total[g]);
        atomicAdd(&out_target_ptr[g], s_target[g]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&out_count_ptr[g]), s_count[g]);
    }
}

__global__ void q8_kernel_global_years(
    const int32_t* __restrict__ year_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ suppkey_ptr,
    const uint8_t* __restrict__ target_supp_ptr,
    int64_t target_supp_len,
    int32_t base_year,
    int32_t n_years,
    double* __restrict__ out_target_ptr,
    double* __restrict__ out_total_ptr,
    int64_t* __restrict__ out_count_ptr,
    int64_t n_elements
) {
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n_elements; idx += stride) {
        int32_t g = year_ptr[idx] - base_year;
        if (g >= 0 && g < n_years) {
            double vol = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            int64_t sk = suppkey_ptr[idx];
            bool is_target = (sk >= 0 && sk < target_supp_len && target_supp_ptr[sk] != 0);
            atomicAdd(&out_total_ptr[g], vol);
            if (is_target) {
                atomicAdd(&out_target_ptr[g], vol);
            }
            atomicAdd(reinterpret_cast<unsigned long long*>(&out_count_ptr[g]), 1ULL);
        }
    }
}

std::vector<torch::Tensor> q8_launcher(
    torch::Tensor years,
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor suppkeys,
    torch::Tensor target_supp_flags,
    int32_t base_year,
    int32_t n_years
) {
    TORCH_CHECK(years.scalar_type() == torch::kInt32, "years must be int32");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(suppkeys.scalar_type() == torch::kInt64, "suppkeys must be int64");
    TORCH_CHECK(target_supp_flags.scalar_type() == torch::kUInt8, "target_supp_flags must be uint8");

    if (!years.is_cuda()) years = years.cuda();
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!suppkeys.is_cuda()) suppkeys = suppkeys.cuda();
    if (!target_supp_flags.is_cuda()) target_supp_flags = target_supp_flags.cuda();

    years = years.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();
    suppkeys = suppkeys.contiguous();
    target_supp_flags = target_supp_flags.contiguous();

    TORCH_CHECK(price.numel() == years.numel(), "price/year length mismatch");
    TORCH_CHECK(discount.numel() == years.numel(), "discount/year length mismatch");
    TORCH_CHECK(suppkeys.numel() == years.numel(), "suppkey/year length mismatch");
    TORCH_CHECK(n_years >= 0, "n_years must be non-negative");

    auto dbl_opts = price.options().dtype(torch::kFloat64);
    auto cnt_opts = years.options().dtype(torch::kInt64);
    auto out_target = torch::zeros({n_years}, dbl_opts);
    auto out_total = torch::zeros({n_years}, dbl_opts);
    auto out_count = torch::zeros({n_years}, cnt_opts);

    int64_t n = years.numel();
    if (n == 0 || n_years == 0) {
        return {out_target, out_total, out_count};
    }

    int blocks = static_cast<int>(std::min<int64_t>((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 4096));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n_years == 2) {
        q8_kernel_two_year<<<blocks, BLOCK_SIZE, 0, stream>>>(
            years.data_ptr<int32_t>(),
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            suppkeys.data_ptr<int64_t>(),
            target_supp_flags.data_ptr<uint8_t>(),
            target_supp_flags.numel(),
            base_year,
            out_target.data_ptr<double>(),
            out_total.data_ptr<double>(),
            out_count.data_ptr<int64_t>(),
            n
        );
    } else if (n_years <= MAX_YEARS) {
        q8_kernel_shared_years<<<blocks, BLOCK_SIZE, 0, stream>>>(
            years.data_ptr<int32_t>(),
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            suppkeys.data_ptr<int64_t>(),
            target_supp_flags.data_ptr<uint8_t>(),
            target_supp_flags.numel(),
            base_year,
            n_years,
            out_target.data_ptr<double>(),
            out_total.data_ptr<double>(),
            out_count.data_ptr<int64_t>(),
            n
        );
    } else {
        q8_kernel_global_years<<<blocks, BLOCK_SIZE, 0, stream>>>(
            years.data_ptr<int32_t>(),
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            suppkeys.data_ptr<int64_t>(),
            target_supp_flags.data_ptr<uint8_t>(),
            target_supp_flags.numel(),
            base_year,
            n_years,
            out_target.data_ptr<double>(),
            out_total.data_ptr<double>(),
            out_count.data_ptr<int64_t>(),
            n
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {out_target, out_total, out_count};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q8_launcher", &q8_launcher, "TPC-H Q8 fixed-year market-share aggregation (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension from the in-module .cu string.

    Runs at Python import time on the host.  The query hot path is in the
    compiled CUDA kernels: the final projection and grouped aggregation are
    fused into one pass over compact joined columns, avoiding PyTorch
    ``unique``/``scatter_add`` temporaries.
    """
    tmpdir = tempfile.mkdtemp(prefix="q8_cuda_")
    cu_path = os.path.join(tmpdir, "_q8_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
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


def _build_supplier_target_flags(
    df_supplier: cudf.DataFrame,
    df_nation: cudf.DataFrame,
    nation_name: str,
) -> torch.Tensor:
    """Build a dense GPU lookup table ``suppkey -> is target nation``.

    cuDF performs the tiny string/nation filtering.  Torch is used only for
    small GPU tensor materialization of the dimension-table lookup, while the
    large fact-table market-share aggregation runs in CUDA.
    """
    max_suppkey = int(df_supplier["s_suppkey"].max())
    target_nation_keys = df_nation[df_nation["n_name"] == nation_name]["n_nationkey"]
    target_suppliers = df_supplier[df_supplier["s_nationkey"].isin(target_nation_keys)][
        "s_suppkey"
    ]

    target_supp_t = torch.as_tensor(target_suppliers.to_cupy(), dtype=torch.int64)
    flags = torch.zeros(
        max_suppkey + 1,
        dtype=torch.uint8,
        device=target_supp_t.device,
    )
    if target_supp_t.numel() != 0:
        flags[target_supp_t] = 1
    return flags


def _query_core(
    o_year: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_suppkey: torch.Tensor,
    target_supp_flags: torch.Tensor,
    base_year: int,
    n_years: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the final Q8 projection and aggregation in CUDA.

    The CUDA kernel scans compact joined columns once, computes
    ``volume = l_extendedprice * (1 - l_discount)`` in registers, looks up the
    supplier-nation flag, and reduces total/target volume by year.  For the
    common two-year Q8 window it uses block reductions with one global atomic
    per block per aggregate, replacing the reference Torch
    ``unique``/``where``/``scatter_add`` sequence.
    """
    if n_years <= 0:
        device = l_extendedprice.device
        return (
            torch.empty((0,), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.float64, device=device),
        )

    target_sum, total_sum, counts = _EXT.q8_launcher(
        o_year,
        l_extendedprice,
        l_discount,
        l_suppkey,
        target_supp_flags,
        int(base_year),
        int(n_years),
    )

    all_years = torch.arange(
        int(base_year),
        int(base_year) + int(n_years),
        dtype=torch.int32,
        device=total_sum.device,
    )
    keep = counts > 0
    mkt_share = target_sum / total_sum
    return all_years[keep], mkt_share[keep]


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
                sum(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END)
                / sum(volume) AS mkt_share
         FROM (
             SELECT extract(year FROM o_orderdate) AS o_year,
                    l_extendedprice * (1 - l_discount) AS volume,
                    n2.n_name AS nation
             FROM part, supplier, lineitem, orders, customer, nation n1,
                  nation n2, region
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

    Description: TPC-H Q8 market share of a supplier nation within a customer
        region for a part type.  cuDF handles joins and string predicates
        (region, part type, nation dimensions).  The final numeric hot path
        runs in a fused hand-written CUDA kernel over compact joined columns:
        volume projection, supplier-nation flag lookup, per-year total volume,
        and per-year target volume are computed in one coalesced scan with
        block-level reductions.  This reduces temporary tensors and replaces
        eager Torch unique/scatter aggregation.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables.
        region_name (str): Region filter for the customer-side nation.
        part_type (str): Part type filter.
        nation_name (str): Supplier nation whose market share is computed.
        orderdate_lo (str): Inclusive order-date lower bound.
        orderdate_hi (str): Inclusive order-date upper bound.
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    lo_ts = pd.to_datetime(orderdate_lo)
    hi_ts = pd.to_datetime(orderdate_hi)
    lo_days = int(lo_ts.timestamp() // 86400)
    hi_days = int(hi_ts.timestamp() // 86400)
    base_year = int(lo_ts.year)
    n_years = max(0, int(hi_ts.year) - base_year + 1)

    region_f = df_region[df_region["r_name"] == region_name][["r_regionkey"]]
    n1 = df_nation.merge(region_f, left_on="n_regionkey", right_on="r_regionkey")[
        ["n_nationkey"]
    ].rename(columns={"n_nationkey": "n1_nationkey"})

    cust_r = df_customer.merge(n1, left_on="c_nationkey", right_on="n1_nationkey")[
        ["c_custkey"]
    ]

    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days <= hi_days)][
        ["o_orderkey", "o_custkey", "o_orderdate"]
    ]
    orders_f["o_year"] = orders_f["o_orderdate"].dt.year.astype("int32")
    orders_f = orders_f.merge(cust_r, left_on="o_custkey", right_on="c_custkey")[
        ["o_orderkey", "o_year"]
    ]

    part_f = df_part[df_part["p_type"] == part_type][["p_partkey"]]

    li = df_lineitem.merge(part_f, left_on="l_partkey", right_on="p_partkey")[
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
    ]

    li = li.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[
        ["l_suppkey", "l_extendedprice", "l_discount", "o_year"]
    ]

    o_year_t = torch.as_tensor(li["o_year"].to_cupy(), dtype=torch.int32)
    suppkey_t = torch.as_tensor(li["l_suppkey"].to_cupy(), dtype=torch.int64)
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64)
    target_supp_flags = _build_supplier_target_flags(df_supplier, df_nation, nation_name)

    years_t, mkt_share_t = _query_core(
        o_year_t,
        price_t,
        discount_t,
        suppkey_t,
        target_supp_flags,
        base_year,
        n_years,
    )

    return cudf.DataFrame(
        {
            "o_year": years_t,
            "mkt_share": mkt_share_t,
        }
    )
