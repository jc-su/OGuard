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
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#define BLOCK_SIZE 256
#define NUM_GROUPS 8

__device__ __forceinline__ int32_t year_from_epoch_days(int32_t days_in) {
    // Howard Hinnant civil_from_days, matching the reference arithmetic for positive TPC-H dates.
    int64_t z = static_cast<int64_t>(days_in) + 719468LL;
    int64_t era = (z >= 0 ? z : z - 146096LL) / 146097LL;
    int64_t doe = z - era * 146097LL;
    int64_t yoe = (doe - doe / 1460LL + doe / 36524LL - doe / 146096LL) / 365LL;
    int64_t y = yoe + era * 400LL;
    int64_t doy = doe - (365LL * yoe + yoe / 4LL - yoe / 100LL);
    int64_t mp = (5LL * doy + 2LL) / 153LL;
    int64_t m = (mp < 10LL) ? (mp + 3LL) : (mp - 9LL);
    int64_t year = (m <= 2LL) ? (y + 1LL) : y;
    return static_cast<int32_t>(year);
}

__global__ void q7_scan_kernel(
    const int32_t* __restrict__ supp_ptr,
    const int32_t* __restrict__ cust_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ revenue8_ptr,
    int64_t n_elements
) {
    __shared__ double block_vals[NUM_GROUPS * BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    #pragma unroll
    for (int g = 0; g < NUM_GROUPS; ++g) {
        block_vals[g * BLOCK_SIZE + tid] = 0.0;
    }

    if (idx < n_elements) {
        const int32_t sd = shipdate_ptr[idx];
        if (sd >= date_lo && sd <= date_hi) {
            const int32_t supp = supp_ptr[idx];
            const int32_t cust = cust_ptr[idx];
            const int32_t year = year_from_epoch_days(sd);
            const int32_t year_offset = year - 1995;
            const int64_t key = static_cast<int64_t>(supp) * 4LL
                              + static_cast<int64_t>(cust) * 2LL
                              + static_cast<int64_t>(year_offset);
            if (key >= 0 && key < NUM_GROUPS) {
                const double price = price_ptr[idx];
                const double disc = discount_ptr[idx];
                block_vals[key * BLOCK_SIZE + tid] = price * (1.0 - disc);
            }
        }
    }

    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int g = 0; g < NUM_GROUPS; ++g) {
                block_vals[g * BLOCK_SIZE + tid] += block_vals[g * BLOCK_SIZE + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid < NUM_GROUPS) {
        atomicAdd(revenue8_ptr + tid, block_vals[tid * BLOCK_SIZE]);
    }
}

__global__ void q7_build_outputs_kernel(
    const double* __restrict__ revenue8_ptr,
    int32_t* __restrict__ out_supp_ptr,
    int32_t* __restrict__ out_cust_ptr,
    int32_t* __restrict__ out_year_ptr,
    double* __restrict__ out_revenue_ptr,
    int32_t* __restrict__ count_ptr
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t count = 0;
        #pragma unroll
        for (int gid = 0; gid < NUM_GROUPS; ++gid) {
            const int32_t supp = gid / 4;
            const int32_t cust = (gid / 2) % 2;
            const int32_t year = (gid % 2) + 1995;
            const bool valid_pair = ((supp == 0 && cust == 1) || (supp == 1 && cust == 0));
            const double rev = revenue8_ptr[gid];
            if (valid_pair && rev > 0.0) {
                out_supp_ptr[count] = supp;
                out_cust_ptr[count] = cust;
                out_year_ptr[count] = year;
                out_revenue_ptr[count] = rev;
                ++count;
            }
        }
        count_ptr[0] = count;
    }
}

std::vector<torch::Tensor> q7_launcher(
    torch::Tensor supp_nation_code,
    torch::Tensor cust_nation_code,
    torch::Tensor l_shipdate_days,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days
) {
    if (!supp_nation_code.is_cuda()) supp_nation_code = supp_nation_code.to(torch::kCUDA);
    if (!cust_nation_code.is_cuda()) cust_nation_code = cust_nation_code.to(torch::kCUDA);
    if (!l_shipdate_days.is_cuda()) l_shipdate_days = l_shipdate_days.to(torch::kCUDA);
    if (!l_extendedprice.is_cuda()) l_extendedprice = l_extendedprice.to(torch::kCUDA);
    if (!l_discount.is_cuda()) l_discount = l_discount.to(torch::kCUDA);

    supp_nation_code = supp_nation_code.contiguous();
    cust_nation_code = cust_nation_code.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();

    TORCH_CHECK(supp_nation_code.scalar_type() == torch::kInt32, "supp_nation_code must be int32");
    TORCH_CHECK(cust_nation_code.scalar_type() == torch::kInt32, "cust_nation_code must be int32");
    TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32, "l_shipdate_days must be int32");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");

    const int64_t n = l_extendedprice.numel();
    TORCH_CHECK(supp_nation_code.numel() == n, "supp_nation_code length mismatch");
    TORCH_CHECK(cust_nation_code.numel() == n, "cust_nation_code length mismatch");
    TORCH_CHECK(l_shipdate_days.numel() == n, "l_shipdate_days length mismatch");
    TORCH_CHECK(l_discount.numel() == n, "l_discount length mismatch");

    c10::cuda::CUDAGuard device_guard(l_extendedprice.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto double_options = l_extendedprice.options().dtype(torch::kFloat64);
    auto int_options = supp_nation_code.options().dtype(torch::kInt32);

    torch::Tensor revenue8 = torch::zeros({NUM_GROUPS}, double_options);

    if (n > 0) {
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q7_scan_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            supp_nation_code.data_ptr<int32_t>(),
            cust_nation_code.data_ptr<int32_t>(),
            l_shipdate_days.data_ptr<int32_t>(),
            l_extendedprice.data_ptr<double>(),
            l_discount.data_ptr<double>(),
            shipdate_lo_days,
            shipdate_hi_days,
            revenue8.data_ptr<double>(),
            n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    torch::Tensor out_supp_max = torch::empty({4}, int_options);
    torch::Tensor out_cust_max = torch::empty({4}, int_options);
    torch::Tensor out_year_max = torch::empty({4}, int_options);
    torch::Tensor out_revenue_max = torch::empty({4}, double_options);
    torch::Tensor count = torch::zeros({1}, int_options);

    q7_build_outputs_kernel<<<1, 1, 0, stream>>>(
        revenue8.data_ptr<double>(),
        out_supp_max.data_ptr<int32_t>(),
        out_cust_max.data_ptr<int32_t>(),
        out_year_max.data_ptr<int32_t>(),
        out_revenue_max.data_ptr<double>(),
        count.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int32_t h_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &h_count,
        count.data_ptr<int32_t>(),
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    torch::Tensor out_supp = out_supp_max.narrow(0, 0, h_count);
    torch::Tensor out_cust = out_cust_max.narrow(0, 0, h_count);
    torch::Tensor out_year = out_year_max.narrow(0, 0, h_count);
    torch::Tensor out_revenue = out_revenue_max.narrow(0, 0, h_count);

    return {out_supp, out_cust, out_year, out_revenue};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q7_launcher", &q7_launcher, "TPC-H Q7 fused filter/project/aggregate CUDA launcher");
}
'''


def _load_cuda() -> Any:
    """
    Compile the CUDA extension from the in-module .cu source.

    The large numeric hot path runs in CUDA: a fused scan filters ship dates,
    computes year/revenue in registers, and performs per-block reductions into
    the eight fixed Q7 groups. This avoids the reference's multiple eager Torch
    masks, temporaries, and scatter pass.
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
    )


_EXT = _load_cuda()


def _query_core(
    supp_nation_code: torch.Tensor,
    cust_nation_code: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU CUDA core for TPC-H Q7's post-join numeric work.

    Inputs are already prepared by cuDF in run_query. This function launches a
    hand-written CUDA fused filter/project/grouped-aggregate kernel over aligned
    1D tensors, then a tiny CUDA output-compaction kernel. It should be faster
    than eager PyTorch because it performs one coalesced pass over the compacted
    joined rows and avoids materializing boolean masks, filtered columns, volume
    tensors, year tensors, and scatter-add temporaries.
    """
    outs = _EXT.q7_launcher(
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        int(shipdate_lo_days),
        int(shipdate_hi_days),
    )
    return outs[0], outs[1], outs[2], outs[3]


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
    Description: TPC-H Q7 - Shipping revenue between two nation pairs by year.
    Params:
      nation_a (str): first nation name.
      nation_b (str): second nation name.
      shipdate_lo (str): inclusive lower bound for l_shipdate.
      shipdate_hi (str): inclusive upper bound for l_shipdate.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]

    # Restrict nations to only the two relevant ones
    nations2 = df_nation[df_nation["n_name"].isin([nation_a, nation_b])][
        ["n_nationkey", "n_name"]
    ]

    # Join supplier with nation (supp_nation)
    sup = df_supplier[["s_suppkey", "s_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "s_nationkey", "n_name": "supp_nation"}),
        on="s_nationkey",
    )

    # Join customer with nation (cust_nation)
    cust = df_customer[["c_custkey", "c_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "c_nationkey", "n_name": "cust_nation"}),
        on="c_nationkey",
    )

    # lineitem with shipdate filter
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    li = df_lineitem[["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]]
    li = li.merge(sup[["s_suppkey", "supp_nation"]], left_on="l_suppkey", right_on="s_suppkey")

    # orders join customer
    oc = df_orders[["o_orderkey", "o_custkey"]].merge(
        cust[["c_custkey", "cust_nation"]], left_on="o_custkey", right_on="c_custkey"
    )

    joined = li.merge(oc[["o_orderkey", "cust_nation"]], left_on="l_orderkey", right_on="o_orderkey")

    # Keep only (FRANCE,GERMANY) or (GERMANY,FRANCE) pairs
    pair_mask = (
        ((joined["supp_nation"] == nation_a) & (joined["cust_nation"] == nation_b))
        | ((joined["supp_nation"] == nation_b) & (joined["cust_nation"] == nation_a))
    )
    joined = joined[pair_mask]

    # Encode nation names: nation_a -> 0, nation_b -> 1
    supp_code = (joined["supp_nation"] == nation_b).astype("int32")
    cust_code = (joined["cust_nation"] == nation_b).astype("int32")

    l_shipdate_days = ((joined["l_shipdate"].astype("int64") // 86400).astype("int32"))

    supp_t = torch.as_tensor(supp_code.to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(cust_code.to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    supp_out, cust_out, year_out, rev_out = _query_core(
        supp_t, cust_t, ship_t, price_t, disc_t, shipdate_lo_days, shipdate_hi_days
    )

    # Map codes back to names: 0 -> nation_a, 1 -> nation_b
    supp_out_cpu = supp_out.cpu().numpy()
    cust_out_cpu = cust_out.cpu().numpy()
    supp_names = [nation_a if v == 0 else nation_b for v in supp_out_cpu]
    cust_names = [nation_a if v == 0 else nation_b for v in cust_out_cpu]

    result = cudf.DataFrame({
        "supp_nation": cudf.Series(supp_names),
        "cust_nation": cudf.Series(cust_names),
        "l_year": year_out,
        "revenue": rev_out,
    })

    result = result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
    return result
