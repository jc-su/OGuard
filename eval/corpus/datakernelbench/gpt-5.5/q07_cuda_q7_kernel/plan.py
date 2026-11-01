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
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024
#define NUM_GROUPS 8

__device__ __forceinline__ int civil_year_from_days(int32_t z_in) {
    // Howard Hinnant civil calendar conversion. Input is days since 1970-01-01.
    int64_t z = static_cast<int64_t>(z_in) + 719468LL;
    int64_t era = (z >= 0 ? z : z - 146096LL) / 146097LL;
    int64_t doe = z - era * 146097LL;
    int64_t yoe = (doe - doe / 1460LL + doe / 36524LL - doe / 146096LL) / 365LL;
    int64_t y = yoe + era * 400LL;
    int64_t doy = doe - (365LL * yoe + yoe / 4LL - yoe / 100LL);
    int64_t mp = (5LL * doy + 2LL) / 153LL;
    int64_t m = (mp < 10LL) ? (mp + 3LL) : (mp - 9LL);
    if (m <= 2LL) {
        y += 1LL;
    }
    return static_cast<int>(y);
}

__global__ void q7_kernel(
    const int32_t* __restrict__ supp_code,
    const int32_t* __restrict__ cust_code,
    const int32_t* __restrict__ shipdate_days,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    int32_t date_lo,
    int32_t date_hi,
    int32_t base_year,
    double* __restrict__ out,
    int64_t n
) {
    __shared__ double block_sums[NUM_GROUPS];

    int tid = threadIdx.x;
    if (tid < NUM_GROUPS) {
        block_sums[tid] = 0.0;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    if (idx < n) {
        int32_t sd = shipdate_days[idx];
        if (sd >= date_lo && sd <= date_hi) {
            int32_t s = supp_code[idx];
            int32_t c = cust_code[idx];

            // SQL keeps only the two directed cross-nation pairs.
            if (((s == 0) && (c == 1)) || ((s == 1) && (c == 0))) {
                int yr = civil_year_from_days(sd);
                int yoff = yr - base_year;

                // Match the reference's fixed 8-key layout:
                // key = supp*4 + cust*2 + (year - 1995), years 1995/1996.
                if (yoff >= 0 && yoff < 2) {
                    int key = s * 4 + c * 2 + yoff;
                    double rev = price[idx] * (1.0 - discount[idx]);
                    atomicAdd(&block_sums[key], rev);
                }
            }
        }
    }

    __syncthreads();

    if (tid < NUM_GROUPS) {
        double v = block_sums[tid];
        if (v != 0.0) {
            atomicAdd(&out[tid], v);
        }
    }
}

torch::Tensor q7_launcher(
    torch::Tensor supp_code,
    torch::Tensor cust_code,
    torch::Tensor shipdate_days,
    torch::Tensor price,
    torch::Tensor discount,
    int32_t date_lo,
    int32_t date_hi,
    int32_t base_year
) {
    if (!supp_code.is_cuda()) supp_code = supp_code.cuda();
    if (!cust_code.is_cuda()) cust_code = cust_code.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();

    supp_code = supp_code.contiguous();
    cust_code = cust_code.contiguous();
    shipdate_days = shipdate_days.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();

    TORCH_CHECK(supp_code.scalar_type() == torch::kInt32, "supp_code must be int32");
    TORCH_CHECK(cust_code.scalar_type() == torch::kInt32, "cust_code must be int32");
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32, "shipdate_days must be int32");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");

    int64_t n = price.numel();
    TORCH_CHECK(supp_code.numel() == n, "supp_code length mismatch");
    TORCH_CHECK(cust_code.numel() == n, "cust_code length mismatch");
    TORCH_CHECK(shipdate_days.numel() == n, "shipdate_days length mismatch");
    TORCH_CHECK(discount.numel() == n, "discount length mismatch");

    auto out = torch::zeros({NUM_GROUPS}, price.options());

    if (n > 0) {
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        q7_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            supp_code.data_ptr<int32_t>(),
            cust_code.data_ptr<int32_t>(),
            shipdate_days.data_ptr<int32_t>(),
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            date_lo,
            date_hi,
            base_year,
            out.data_ptr<double>(),
            n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q7_launcher", &q7_launcher, "TPC-H Q7 fused revenue aggregation (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension from the embedded .cu source.

    The compiled code runs the memory-bound post-join filter/project/aggregate on
    the GPU in one fused pass, avoiding the reference plan's multiple eager Torch
    masks, temporary tensors, and scatter-add setup.
    """
    tmpdir = tempfile.mkdtemp(prefix="q7_cuda_")
    cu_path = os.path.join(tmpdir, "_q7_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q7_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _empty_result() -> cudf.DataFrame:
    """Create an empty cuDF result with the SQL output schema on the host/GPU."""
    return cudf.DataFrame(
        {
            "supp_nation": cudf.Series([], dtype="str"),
            "cust_nation": cudf.Series([], dtype="str"),
            "l_year": cudf.Series([], dtype="int32"),
            "revenue": cudf.Series([], dtype="float64"),
        }
    )


def _nation_key(df_nation: cudf.DataFrame, nation_name: str) -> int | None:
    """Resolve one nation name using cuDF string filtering over the tiny nation table."""
    s = df_nation.loc[df_nation["n_name"] == nation_name, "n_nationkey"]
    if len(s) == 0:
        return None
    return int(s.iloc[0])


def _query_core(
    supp_nation_code: torch.Tensor,
    cust_nation_code: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """Run the fused CUDA hot path.

    Inputs are compact post-join GPU tensors. CUDA performs the date/pair checks,
    year extraction, projection ``price * (1 - discount)``, and fixed 8-bin
    aggregation in one coalesced scan. Per-block shared-memory accumulation keeps
    global atomics to at most eight per block.
    """
    return _EXT.q7_launcher(
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        int(shipdate_lo_days),
        int(shipdate_hi_days),
        1995,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_a: str = "FRANCE",
    nation_b: str = "GERMANY",
    shipdate_lo: str = "1995-01-01",
    shipdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL: SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue FROM (
         SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
                extract(year FROM l_shipdate) AS l_year,
                l_extendedprice * (1 - l_discount) AS volume
         FROM supplier, lineitem, orders, customer, nation n1, nation n2
         WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey
               AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey
               AND c_nationkey = n2.n_nationkey
               AND ((n1.n_name='FRANCE' AND n2.n_name='GERMANY')
                 OR (n1.n_name='GERMANY' AND n2.n_name='FRANCE'))
               AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
         ) GROUP BY supp_nation, cust_nation, l_year
         ORDER BY supp_nation, cust_nation, l_year;

    Description: TPC-H Q7 shipping revenue between two nation pairs by year.
      cuDF handles joins, strings, and dimension-table filtering. The bulk
      post-join numeric path runs in a hand-written CUDA kernel that fuses
      filter, year projection, revenue computation, and 8-group aggregation.
      This is faster for the memory-bound scan because it avoids eager Torch
      boolean-mask materialization, several temporary columns, and per-row
      global atomics.

    Params:
      tables (dict[str, cudf.DataFrame]): input TPC-H tables.
      nation_a (str): first nation name, encoded as code 0.
      nation_b (str): second nation name, encoded as code 1.
      shipdate_lo (str): inclusive lower bound for l_shipdate.
      shipdate_hi (str): inclusive upper bound for l_shipdate.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]

    nation_a_key = _nation_key(df_nation, nation_a)
    nation_b_key = _nation_key(df_nation, nation_b)
    if nation_a_key is None or nation_b_key is None:
        return _empty_result()

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    sup = df_supplier.loc[
        df_supplier["s_nationkey"].isin([nation_a_key, nation_b_key]),
        ["s_suppkey", "s_nationkey"],
    ].copy(deep=False)
    if len(sup) == 0:
        return _empty_result()
    sup["supp_code"] = (sup["s_nationkey"] == nation_b_key).astype("int32")
    sup = sup[["s_suppkey", "supp_code"]]

    cust = df_customer.loc[
        df_customer["c_nationkey"].isin([nation_a_key, nation_b_key]),
        ["c_custkey", "c_nationkey"],
    ].copy(deep=False)
    if len(cust) == 0:
        return _empty_result()
    cust["cust_code"] = (cust["c_nationkey"] == nation_b_key).astype("int32")
    cust = cust[["c_custkey", "cust_code"]]

    oc = df_orders[["o_orderkey", "o_custkey"]].merge(
        cust,
        left_on="o_custkey",
        right_on="c_custkey",
    )[["o_orderkey", "cust_code"]]
    if len(oc) == 0:
        return _empty_result()

    l_shipdate_days_all = (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")
    date_mask = (l_shipdate_days_all >= shipdate_lo_days) & (l_shipdate_days_all <= shipdate_hi_days)

    li = df_lineitem.loc[
        date_mask,
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"],
    ].copy(deep=False)
    if len(li) == 0:
        return _empty_result()
    li["l_shipdate_days"] = l_shipdate_days_all[date_mask]

    joined = li.merge(sup, left_on="l_suppkey", right_on="s_suppkey")[
        ["l_orderkey", "l_shipdate_days", "l_extendedprice", "l_discount", "supp_code"]
    ]
    if len(joined) == 0:
        return _empty_result()

    joined = joined.merge(oc, left_on="l_orderkey", right_on="o_orderkey")[
        ["supp_code", "cust_code", "l_shipdate_days", "l_extendedprice", "l_discount"]
    ]
    if len(joined) == 0:
        return _empty_result()

    supp_t = torch.as_tensor(joined["supp_code"].to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(joined["cust_code"].to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(joined["l_shipdate_days"].to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    revenue_t = _query_core(
        supp_t,
        cust_t,
        ship_t,
        price_t,
        disc_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    revenue_cpu = revenue_t.detach().cpu().numpy()

    supp_names: list[str] = []
    cust_names: list[str] = []
    years: list[int] = []
    revenues: list[float] = []

    for gid in range(8):
        s_code = gid // 4
        c_code = (gid // 2) % 2
        year = 1995 + (gid % 2)
        rev = float(revenue_cpu[gid])
        valid_pair = ((s_code == 0 and c_code == 1) or (s_code == 1 and c_code == 0))
        if valid_pair and rev > 0.0:
            supp_names.append(nation_a if s_code == 0 else nation_b)
            cust_names.append(nation_a if c_code == 0 else nation_b)
            years.append(year)
            revenues.append(rev)

    if not revenues:
        return _empty_result()

    result = cudf.DataFrame(
        {
            "supp_nation": cudf.Series(supp_names),
            "cust_nation": cudf.Series(cust_names),
            "l_year": cudf.Series(years, dtype="int32"),
            "revenue": cudf.Series(revenues, dtype="float64"),
        }
    )
    return result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
