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

#include <algorithm>
#include <cstdint>
#include <vector>

#define Q4_BLOCK_SIZE 256

__global__ void q4_mark_late_orders_kernel(
    const int64_t* __restrict__ l_orderkey,
    const int32_t* __restrict__ l_commitdate_days,
    const int32_t* __restrict__ l_receiptdate_days,
    uint8_t* __restrict__ has_late_lineitem,
    int64_t n_lineitem,
    int64_t max_key
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_lineitem) {
        int32_t commit_day = l_commitdate_days[idx];
        int32_t receipt_day = l_receiptdate_days[idx];
        if (commit_day < receipt_day) {
            int64_t key = l_orderkey[idx];
            if (key >= 0 && key <= max_key) {
                has_late_lineitem[key] = static_cast<uint8_t>(1);
            }
        }
    }
}

__global__ void q4_count_priorities_kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_orderdate_days,
    const int32_t* __restrict__ o_orderpriority_codes,
    const uint8_t* __restrict__ has_late_lineitem,
    int64_t* __restrict__ counts,
    int64_t n_orders,
    int64_t max_key,
    int32_t date_lo_days,
    int32_t date_hi_days,
    int32_t num_priorities
) {
    extern __shared__ unsigned long long s_counts[];

    for (int32_t p = threadIdx.x; p < num_priorities; p += blockDim.x) {
        s_counts[p] = 0ULL;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int32_t od = o_orderdate_days[idx];
        if (od >= date_lo_days && od < date_hi_days) {
            int64_t key = o_orderkey[idx];
            if (key >= 0 && key <= max_key && has_late_lineitem[key] != 0) {
                int32_t prio = o_orderpriority_codes[idx];
                if (prio >= 0 && prio < num_priorities) {
                    atomicAdd(&s_counts[prio], 1ULL);
                }
            }
        }
    }

    __syncthreads();

    for (int32_t p = threadIdx.x; p < num_priorities; p += blockDim.x) {
        unsigned long long v = s_counts[p];
        if (v != 0ULL) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(counts + p),
                v
            );
        }
    }
}

std::vector<torch::Tensor> q4_launcher(
    torch::Tensor o_orderkey,
    torch::Tensor o_orderdate_days,
    torch::Tensor o_orderpriority_codes,
    torch::Tensor l_orderkey,
    torch::Tensor l_commitdate_days,
    torch::Tensor l_receiptdate_days,
    int32_t date_lo_days,
    int32_t date_hi_days,
    int32_t num_priorities
) {
    TORCH_CHECK(o_orderkey.is_cuda(), "o_orderkey must be a CUDA tensor");
    TORCH_CHECK(o_orderdate_days.is_cuda(), "o_orderdate_days must be a CUDA tensor");
    TORCH_CHECK(o_orderpriority_codes.is_cuda(), "o_orderpriority_codes must be a CUDA tensor");
    TORCH_CHECK(l_orderkey.is_cuda(), "l_orderkey must be a CUDA tensor");
    TORCH_CHECK(l_commitdate_days.is_cuda(), "l_commitdate_days must be a CUDA tensor");
    TORCH_CHECK(l_receiptdate_days.is_cuda(), "l_receiptdate_days must be a CUDA tensor");

    TORCH_CHECK(o_orderkey.scalar_type() == torch::kInt64, "o_orderkey must be int64");
    TORCH_CHECK(o_orderdate_days.scalar_type() == torch::kInt32, "o_orderdate_days must be int32 epoch days");
    TORCH_CHECK(o_orderpriority_codes.scalar_type() == torch::kInt32, "o_orderpriority_codes must be int32");
    TORCH_CHECK(l_orderkey.scalar_type() == torch::kInt64, "l_orderkey must be int64");
    TORCH_CHECK(l_commitdate_days.scalar_type() == torch::kInt32, "l_commitdate_days must be int32 epoch days");
    TORCH_CHECK(l_receiptdate_days.scalar_type() == torch::kInt32, "l_receiptdate_days must be int32 epoch days");
    TORCH_CHECK(num_priorities >= 0, "num_priorities must be non-negative");

    c10::cuda::CUDAGuard device_guard(o_orderkey.device());

    o_orderkey = o_orderkey.contiguous();
    o_orderdate_days = o_orderdate_days.contiguous();
    o_orderpriority_codes = o_orderpriority_codes.contiguous();
    l_orderkey = l_orderkey.contiguous();
    l_commitdate_days = l_commitdate_days.contiguous();
    l_receiptdate_days = l_receiptdate_days.contiguous();

    int64_t n_orders = o_orderkey.numel();
    int64_t n_lineitem = l_orderkey.numel();

    auto out_options = o_orderkey.options().dtype(torch::kInt64);
    auto counts = torch::zeros({static_cast<int64_t>(num_priorities)}, out_options);

    if (num_priorities == 0) {
        auto empty_ids = torch::empty({0}, out_options);
        return {empty_ids, counts};
    }

    if (n_orders == 0 || n_lineitem == 0) {
        auto priority_ids = torch::arange(num_priorities, out_options);
        auto mask = counts > 0;
        return {priority_ids.masked_select(mask), counts.masked_select(mask)};
    }

    int64_t max_o_key = torch::max(o_orderkey).item<int64_t>();
    int64_t max_l_key = torch::max(l_orderkey).item<int64_t>();
    int64_t max_key = std::max(max_o_key, max_l_key);
    TORCH_CHECK(max_key >= 0, "order keys must be non-negative");

    auto marker_options = o_orderkey.options().dtype(torch::kUInt8);
    auto has_late_lineitem = torch::zeros({max_key + 1}, marker_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int blocks_l = static_cast<int>((n_lineitem + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE);
    q4_mark_late_orders_kernel<<<blocks_l, Q4_BLOCK_SIZE, 0, stream>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_commitdate_days.data_ptr<int32_t>(),
        l_receiptdate_days.data_ptr<int32_t>(),
        has_late_lineitem.data_ptr<uint8_t>(),
        n_lineitem,
        max_key
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_o = static_cast<int>((n_orders + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE);
    size_t shared_bytes = static_cast<size_t>(num_priorities) * sizeof(unsigned long long);
    q4_count_priorities_kernel<<<blocks_o, Q4_BLOCK_SIZE, shared_bytes, stream>>>(
        o_orderkey.data_ptr<int64_t>(),
        o_orderdate_days.data_ptr<int32_t>(),
        o_orderpriority_codes.data_ptr<int32_t>(),
        has_late_lineitem.data_ptr<uint8_t>(),
        counts.data_ptr<int64_t>(),
        n_orders,
        max_key,
        date_lo_days,
        date_hi_days,
        num_priorities
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto priority_ids = torch::arange(num_priorities, out_options);
    auto mask = counts > 0;
    auto priority_ids_nz = priority_ids.masked_select(mask);
    auto counts_nz = counts.masked_select(mask);
    return {priority_ids_nz, counts_nz};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q4_launcher", &q4_launcher, "TPC-H Q4 EXISTS/count by priority using fused CUDA scans");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension once from an in-module source string.

    The hot path runs in handwritten CUDA: one pass marks orderkeys with any
    late lineitem, and one pass filters orders by date plus that marker and
    accumulates the small priority histogram. This avoids the reference
    torch.unique/torch.isin materializations and replaces several memory-bound
    tensor passes with two coalesced CUDA scans.
    """
    tmp_root = os.path.join(tempfile.gettempdir(), "q4_cuda_ext_sm90")
    build_dir = os.path.join(tmp_root, "build")
    cu_path = os.path.join(tmp_root, "_q4_kernels.cu")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q4_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _query_core(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TPC-H Q4's numeric EXISTS and GROUP BY hot path on the GPU.

    CUDA marks each orderkey that has at least one lineitem with
    l_commitdate < l_receiptdate, then scans orders once to apply the date
    predicate and update a small per-priority histogram. This is faster than
    the eager PyTorch plan for this memory-bound query because it avoids
    torch.unique/torch.isin sorting/probing temporaries and uses fused,
    coalesced scans with block-local histogram aggregation.
    """
    result = _EXT.q4_launcher(
        o_orderkey,
        o_orderdate_days,
        o_orderpriority_codes,
        l_orderkey,
        l_commitdate_days,
        l_receiptdate_days,
        int(date_lo_days),
        int(date_hi_days),
        int(num_priorities),
    )
    return result[0], result[1]


def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count FROM orders WHERE o_orderdate >= CAST('1993-07-01' AS date) AND o_orderdate < CAST('1993-10-01' AS date) AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority;
    Description: TPC-H Q4 - Counts orders per priority within a date range where at least one lineitem shipped late (commit before receipt).
    Params:
        date_lo (str): inclusive lower bound for o_orderdate.
        date_hi (str): exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    # Encode o_orderpriority as categorical for groupby on GPU
    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes  # int codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)
    o_orderdate_days_t = torch.as_tensor(
        ((df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    o_priority_codes_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32)

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_commitdate_days_t = torch.as_tensor(
        ((df_lineitem["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_receiptdate_days_t = torch.as_tensor(
        ((df_lineitem["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    priority_ids_t, counts_t = _query_core(
        o_orderkey_t,
        o_orderdate_days_t,
        o_priority_codes_t,
        l_orderkey_t,
        l_commitdate_days_t,
        l_receiptdate_days_t,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )

    # Map codes back to priority strings and sort
    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[i] for i in priority_ids_host]

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
        "order_count": counts_t,
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
