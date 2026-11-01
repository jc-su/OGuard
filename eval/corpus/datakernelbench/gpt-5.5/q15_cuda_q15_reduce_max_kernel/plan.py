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
#include <stdint.h>
#include <cfloat>
#include <vector>

#define BLOCK_SIZE 1024
#define EMPTY_KEY (-1LL)

__device__ __forceinline__ uint64_t q15_hash_i64(int64_t key) {
    uint64_t x = static_cast<uint64_t>(key);
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ int64_t q15_lookup_supplier(
    int64_t key,
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    int64_t hash_mask
) {
    uint64_t h = q15_hash_i64(key);
    int64_t slot = static_cast<int64_t>(h & static_cast<uint64_t>(hash_mask));

    #pragma unroll 1
    for (int64_t probe = 0; probe <= hash_mask; ++probe) {
        int64_t cur = hash_keys[slot];
        if (cur == key) {
            return hash_vals[slot];
        }
        if (cur == EMPTY_KEY) {
            return -1;
        }
        slot = (slot + 1) & hash_mask;
    }
    return -1;
}

__global__ void q15_build_hash_kernel(
    const int64_t* __restrict__ s_suppkey,
    int64_t* __restrict__ hash_keys,
    int64_t* __restrict__ hash_vals,
    int64_t n_sup,
    int64_t hash_mask
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_sup) {
        return;
    }

    int64_t key = s_suppkey[idx];
    uint64_t h = q15_hash_i64(key);
    int64_t slot = static_cast<int64_t>(h & static_cast<uint64_t>(hash_mask));

    #pragma unroll 1
    while (true) {
        unsigned long long* slot_ptr =
            reinterpret_cast<unsigned long long*>(&hash_keys[slot]);
        unsigned long long old = atomicCAS(
            slot_ptr,
            static_cast<unsigned long long>(EMPTY_KEY),
            static_cast<unsigned long long>(key)
        );
        int64_t old_i = static_cast<int64_t>(old);
        if (old_i == EMPTY_KEY || old_i == key) {
            hash_vals[slot] = idx;
            return;
        }
        slot = (slot + 1) & hash_mask;
    }
}

__global__ void q15_aggregate_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    int64_t hash_mask,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ totals,
    uint8_t* __restrict__ present,
    int64_t n_line
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_line) {
        return;
    }

    int32_t shipdate = l_shipdate_days[idx];
    if (shipdate >= shipdate_lo_days && shipdate < shipdate_hi_days) {
        int64_t suppkey = l_suppkey[idx];
        int64_t supplier_idx = q15_lookup_supplier(suppkey, hash_keys, hash_vals, hash_mask);
        if (supplier_idx >= 0) {
            double revenue = l_extendedprice[idx] * (1.0 - l_discount[idx]);
            atomicAdd(&totals[supplier_idx], revenue);
            present[supplier_idx] = static_cast<uint8_t>(1);
        }
    }
}

__global__ void q15_reduce_max_kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    double* __restrict__ block_max,
    int64_t n_sup
) {
    __shared__ double smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    double local_max = -DBL_MAX;
    for (int64_t i = idx; i < n_sup; i += stride) {
        if (present[i]) {
            double v = totals[i];
            local_max = v > local_max ? v : local_max;
        }
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double other = smem[tid + s];
            smem[tid] = other > smem[tid] ? other : smem[tid];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}

__global__ void q15_reduce_final_max_kernel(
    const double* __restrict__ block_max,
    double* __restrict__ max_out,
    int64_t n_blocks
) {
    __shared__ double smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    double local_max = -DBL_MAX;
    for (int64_t i = tid; i < n_blocks; i += blockDim.x) {
        double v = block_max[i];
        local_max = v > local_max ? v : local_max;
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double other = smem[tid + s];
            smem[tid] = other > smem[tid] ? other : smem[tid];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_out[0] = smem[0];
    }
}

__global__ void q15_count_selected_kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    const double* __restrict__ max_value,
    int64_t* __restrict__ count_out,
    int64_t n_sup
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_sup) {
        return;
    }

    double mx = max_value[0];
    if (present[idx] && totals[idx] == mx) {
        atomicAdd(reinterpret_cast<unsigned long long*>(count_out), 1ULL);
    }
}

__global__ void q15_fill_selected_kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    const double* __restrict__ max_value,
    int64_t* __restrict__ counter,
    int64_t* __restrict__ selected_indices,
    double* __restrict__ selected_revenues,
    int64_t n_sup
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_sup) {
        return;
    }

    double mx = max_value[0];
    if (present[idx] && totals[idx] == mx) {
        unsigned long long pos = atomicAdd(
            reinterpret_cast<unsigned long long*>(counter),
            1ULL
        );
        selected_indices[static_cast<int64_t>(pos)] = idx;
        selected_revenues[static_cast<int64_t>(pos)] = totals[idx];
    }
}

std::vector<torch::Tensor> q15_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    torch::Tensor s_suppkey,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days
) {
    TORCH_CHECK(l_suppkey.is_cuda(), "l_suppkey must be a CUDA tensor");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be a CUDA tensor");
    TORCH_CHECK(l_discount.is_cuda(), "l_discount must be a CUDA tensor");
    TORCH_CHECK(l_shipdate_days.is_cuda(), "l_shipdate_days must be a CUDA tensor");
    TORCH_CHECK(s_suppkey.is_cuda(), "s_suppkey must be a CUDA tensor");

    TORCH_CHECK(l_suppkey.scalar_type() == torch::kInt64, "l_suppkey must be int64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32, "l_shipdate_days must be int32 epoch days");
    TORCH_CHECK(s_suppkey.scalar_type() == torch::kInt64, "s_suppkey must be int64");

    l_suppkey = l_suppkey.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();
    s_suppkey = s_suppkey.contiguous();

    int64_t n_line = l_suppkey.numel();
    int64_t n_sup = s_suppkey.numel();

    TORCH_CHECK(l_extendedprice.numel() == n_line, "lineitem tensor size mismatch");
    TORCH_CHECK(l_discount.numel() == n_line, "lineitem tensor size mismatch");
    TORCH_CHECK(l_shipdate_days.numel() == n_line, "lineitem tensor size mismatch");

    auto opts_i64 = s_suppkey.options().dtype(torch::kInt64);
    auto opts_u8 = s_suppkey.options().dtype(torch::kUInt8);
    auto opts_f64 = l_extendedprice.options().dtype(torch::kFloat64);

    torch::Tensor totals = torch::zeros({n_sup}, opts_f64);
    torch::Tensor present = torch::zeros({n_sup}, opts_u8);

    int64_t hash_size = 1;
    while (hash_size < n_sup * 2) {
        hash_size <<= 1;
    }
    int64_t hash_mask = hash_size - 1;

    torch::Tensor hash_keys = torch::full({hash_size}, EMPTY_KEY, opts_i64);
    torch::Tensor hash_vals = torch::empty({hash_size}, opts_i64);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int build_blocks = static_cast<int>((n_sup + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (build_blocks > 0) {
        q15_build_hash_kernel<<<build_blocks, BLOCK_SIZE, 0, stream>>>(
            s_suppkey.data_ptr<int64_t>(),
            hash_keys.data_ptr<int64_t>(),
            hash_vals.data_ptr<int64_t>(),
            n_sup,
            hash_mask
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int agg_blocks = static_cast<int>((n_line + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (agg_blocks > 0) {
        q15_aggregate_kernel<<<agg_blocks, BLOCK_SIZE, 0, stream>>>(
            l_suppkey.data_ptr<int64_t>(),
            l_extendedprice.data_ptr<double>(),
            l_discount.data_ptr<double>(),
            l_shipdate_days.data_ptr<int32_t>(),
            hash_keys.data_ptr<int64_t>(),
            hash_vals.data_ptr<int64_t>(),
            hash_mask,
            shipdate_lo_days,
            shipdate_hi_days,
            totals.data_ptr<double>(),
            present.data_ptr<uint8_t>(),
            n_line
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int reduce_blocks = static_cast<int>((n_sup + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (reduce_blocks < 1) {
        reduce_blocks = 1;
    }
    if (reduce_blocks > 4096) {
        reduce_blocks = 4096;
    }

    torch::Tensor block_max = torch::empty({reduce_blocks}, opts_f64);
    torch::Tensor max_value = torch::empty({1}, opts_f64);

    q15_reduce_max_kernel<<<reduce_blocks, BLOCK_SIZE, 0, stream>>>(
        totals.data_ptr<double>(),
        present.data_ptr<uint8_t>(),
        block_max.data_ptr<double>(),
        n_sup
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    q15_reduce_final_max_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        block_max.data_ptr<double>(),
        max_value.data_ptr<double>(),
        reduce_blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    torch::Tensor count_tensor = torch::zeros({1}, opts_i64);
    int count_blocks = static_cast<int>((n_sup + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (count_blocks > 0) {
        q15_count_selected_kernel<<<count_blocks, BLOCK_SIZE, 0, stream>>>(
            totals.data_ptr<double>(),
            present.data_ptr<uint8_t>(),
            max_value.data_ptr<double>(),
            count_tensor.data_ptr<int64_t>(),
            n_sup
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int64_t selected_count = count_tensor.cpu().item<int64_t>();

    torch::Tensor selected_indices = torch::empty({selected_count}, opts_i64);
    torch::Tensor selected_revenues = torch::empty({selected_count}, opts_f64);
    torch::Tensor fill_counter = torch::zeros({1}, opts_i64);

    if (selected_count > 0 && count_blocks > 0) {
        q15_fill_selected_kernel<<<count_blocks, BLOCK_SIZE, 0, stream>>>(
            totals.data_ptr<double>(),
            present.data_ptr<uint8_t>(),
            max_value.data_ptr<double>(),
            fill_counter.data_ptr<int64_t>(),
            selected_indices.data_ptr<int64_t>(),
            selected_revenues.data_ptr<double>(),
            n_sup
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {selected_indices, selected_revenues};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", &q15_launcher, "TPC-H Q15 fused filter/project/group/max/select CUDA launcher");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension once; all hot-path GPU kernels live in _CUDA_SOURCE."""
    tmpdir = tempfile.mkdtemp(prefix="q15_cuda_")
    cu_path = os.path.join(tmpdir, "_q15_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q15_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _query_core(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the numeric hot path in custom CUDA.

    The CUDA extension builds a small supplier-key hash table, scans lineitem once with
    coalesced date loads, conditionally loads revenue columns only for qualifying dates,
    atomically aggregates fp64 revenue per supplier, reduces the maximum revenue on GPU,
    and returns supplier row indices/revenues for the max supplier(s). This avoids the
    reference plan's materialized boolean mask, filtered tensors, searchsorted, and
    scatter temporaries over the large lineitem table.
    """
    selected_indices, selected_revenues = _EXT.q15_launcher(
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        s_suppkey,
        int(shipdate_lo_days),
        int(shipdate_hi_days),
    )
    torch.cuda.current_stream(selected_indices.device).synchronize()
    return selected_indices, selected_revenues


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1996-01-01",
    shipdate_hi: str = "1996-04-01",
) -> cudf.DataFrame:
    """
    SQL:
      WITH revenue AS (
        SELECT l_suppkey AS supplier_no,
               SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
        GROUP BY l_suppkey
      )
      SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
      FROM supplier, revenue
      WHERE s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
      ORDER BY s_suppkey;

    Description:
      TPC-H Q15 — find supplier(s) with the top revenue in a date window. cuDF is used
      for table access, datetime-to-epoch-day conversion, and final string-column gather.
      The memory-bound lineitem filter/project/group-by/max path runs in fused CUDA:
      one coalesced scan over lineitem, a small GPU hash table for supplier membership,
      fp64 per-supplier revenue aggregation, and GPU max/selection. This reduces large
      intermediate tensors and Python/cuDF/PyTorch round trips compared with the
      reference eager Torch pipeline.

    Params:
      tables (dict[str, cudf.DataFrame]): input TPC-H tables containing lineitem and supplier.
      shipdate_lo (str): inclusive lower shipdate bound.
      shipdate_hi (str): exclusive upper shipdate bound.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_price_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_disc_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)

    sel_idx_t, sel_rev_t = _query_core(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        s_suppkey_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    result = cudf.DataFrame(
        {
            "s_suppkey": supplier_sel["s_suppkey"],
            "s_name": supplier_sel["s_name"],
            "s_address": supplier_sel["s_address"],
            "s_phone": supplier_sel["s_phone"],
            "total_revenue": cudf.Series(sel_rev_t),
        }
    )

    return result.sort_values("s_suppkey").reset_index(drop=True)
