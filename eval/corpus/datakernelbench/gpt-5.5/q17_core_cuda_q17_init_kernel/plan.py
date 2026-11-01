import os
import tempfile
from typing import Any

import torch
import cudf
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <stdint.h>
#include <limits.h>
#include <algorithm>

#define BLOCK_SIZE 1024
#define EMPTY_KEY LLONG_MIN

__device__ __forceinline__ uint64_t q17_hash_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ int64_t q17_lookup_key(
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    int64_t hash_size,
    int64_t key
) {
    uint64_t mask = static_cast<uint64_t>(hash_size - 1);
    uint64_t pos = q17_hash_u64(static_cast<uint64_t>(key)) & mask;

    #pragma unroll 1
    for (int64_t probe = 0; probe < hash_size; ++probe) {
        int64_t found = hash_keys[pos];
        if (found == key) {
            return hash_vals[pos];
        }
        if (found == EMPTY_KEY) {
            return -1;
        }
        pos = (pos + 1ULL) & mask;
    }
    return -1;
}

__global__ void q17_init_kernel(
    int64_t* __restrict__ hash_keys,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    double* __restrict__ out,
    int64_t hash_size,
    int64_t n_parts,
    int64_t n_init
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = tid; i < n_init; i += stride) {
        if (i < hash_size) {
            hash_keys[i] = EMPTY_KEY;
        }
        if (i < n_parts) {
            sum_qty[i] = 0.0;
            cnt_qty[i] = 0.0;
        }
        if (i == 0) {
            *out = 0.0;
        }
    }
}

__global__ void q17_build_hash_kernel(
    const int64_t* __restrict__ part_keys,
    int64_t* __restrict__ hash_keys,
    int64_t* __restrict__ hash_vals,
    int64_t n_parts,
    int64_t hash_size
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    uint64_t mask = static_cast<uint64_t>(hash_size - 1);
    const unsigned long long empty = static_cast<unsigned long long>(EMPTY_KEY);

    for (int64_t idx = tid; idx < n_parts; idx += stride) {
        int64_t key = part_keys[idx];
        uint64_t pos = q17_hash_u64(static_cast<uint64_t>(key)) & mask;

        #pragma unroll 1
        while (true) {
            unsigned long long old = atomicCAS(
                reinterpret_cast<unsigned long long*>(&hash_keys[pos]),
                empty,
                static_cast<unsigned long long>(key)
            );
            int64_t old_signed = static_cast<int64_t>(old);
            if (old_signed == EMPTY_KEY || old_signed == key) {
                hash_vals[pos] = idx;
                break;
            }
            pos = (pos + 1ULL) & mask;
        }
    }
}

__global__ void q17_accumulate_avg_kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n_line,
    int64_t hash_size
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t row = tid; row < n_line; row += stride) {
        int64_t part_idx = q17_lookup_key(hash_keys, hash_vals, hash_size, l_partkey[row]);
        if (part_idx >= 0) {
            atomicAdd(&sum_qty[part_idx], l_quantity[row]);
            atomicAdd(&cnt_qty[part_idx], 1.0);
        }
    }
}

__global__ void q17_final_sum_kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    const double* __restrict__ sum_qty,
    const double* __restrict__ cnt_qty,
    double* __restrict__ out,
    int64_t n_line,
    int64_t hash_size
) {
    __shared__ double block_sum[BLOCK_SIZE];

    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    double local = 0.0;

    for (int64_t row = tid; row < n_line; row += stride) {
        int64_t part_idx = q17_lookup_key(hash_keys, hash_vals, hash_size, l_partkey[row]);
        if (part_idx >= 0) {
            double cnt = cnt_qty[part_idx];
            double sum = sum_qty[part_idx];
            double qty = l_quantity[row];

            if (cnt > 0.0 && qty * cnt < 0.2 * sum) {
                local += l_extendedprice[row];
            }
        }
    }

    block_sum[threadIdx.x] = local;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}

torch::Tensor q17_launcher(
    torch::Tensor l_partkey,
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor part_keys
) {
    TORCH_CHECK(l_partkey.is_cuda(), "l_partkey must be a CUDA tensor");
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be a CUDA tensor");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be a CUDA tensor");
    TORCH_CHECK(part_keys.is_cuda(), "part_keys must be a CUDA tensor");

    TORCH_CHECK(l_partkey.scalar_type() == torch::kInt64, "l_partkey must be int64");
    TORCH_CHECK(part_keys.scalar_type() == torch::kInt64, "part_keys must be int64");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");

    TORCH_CHECK(l_partkey.dim() == 1, "l_partkey must be 1D");
    TORCH_CHECK(l_quantity.dim() == 1, "l_quantity must be 1D");
    TORCH_CHECK(l_extendedprice.dim() == 1, "l_extendedprice must be 1D");
    TORCH_CHECK(part_keys.dim() == 1, "part_keys must be 1D");
    TORCH_CHECK(l_partkey.numel() == l_quantity.numel(), "lineitem tensors must have equal length");
    TORCH_CHECK(l_partkey.numel() == l_extendedprice.numel(), "lineitem tensors must have equal length");

    l_partkey = l_partkey.contiguous();
    l_quantity = l_quantity.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    part_keys = part_keys.contiguous();

    int64_t n_line = l_partkey.numel();
    int64_t n_parts = part_keys.numel();

    auto out = torch::empty({}, l_quantity.options());
    if (n_line == 0 || n_parts == 0) {
        out.zero_();
        return out;
    }

    int64_t hash_size = 1;
    while (hash_size < n_parts * 4) {
        hash_size <<= 1;
    }
    if (hash_size < 2) {
        hash_size = 2;
    }

    auto long_opts = part_keys.options().dtype(torch::kInt64);
    auto dbl_opts = l_quantity.options().dtype(torch::kFloat64);

    auto hash_keys = torch::empty({hash_size}, long_opts);
    auto hash_vals = torch::empty({hash_size}, long_opts);
    auto sum_qty = torch::empty({n_parts}, dbl_opts);
    auto cnt_qty = torch::empty({n_parts}, dbl_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int threads = BLOCK_SIZE;
    int64_t n_init = std::max(hash_size, n_parts);
    int init_blocks = static_cast<int>(std::min<int64_t>((n_init + threads - 1) / threads, 8192));
    if (init_blocks < 1) {
        init_blocks = 1;
    }

    q17_init_kernel<<<init_blocks, threads, 0, stream>>>(
        hash_keys.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        out.data_ptr<double>(),
        hash_size,
        n_parts,
        n_init
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int build_blocks = static_cast<int>(std::min<int64_t>((n_parts + threads - 1) / threads, 8192));
    if (build_blocks < 1) {
        build_blocks = 1;
    }

    q17_build_hash_kernel<<<build_blocks, threads, 0, stream>>>(
        part_keys.data_ptr<int64_t>(),
        hash_keys.data_ptr<int64_t>(),
        hash_vals.data_ptr<int64_t>(),
        n_parts,
        hash_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int scan_blocks = static_cast<int>(std::min<int64_t>((n_line + threads - 1) / threads, 8192));
    if (scan_blocks < 1) {
        scan_blocks = 1;
    }

    q17_accumulate_avg_kernel<<<scan_blocks, threads, 0, stream>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        hash_keys.data_ptr<int64_t>(),
        hash_vals.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        n_line,
        hash_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    q17_final_sum_kernel<<<scan_blocks, threads, 0, stream>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        hash_keys.data_ptr<int64_t>(),
        hash_vals.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        out.data_ptr<double>(),
        n_line,
        hash_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    out.mul_(1.0 / 7.0);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_launcher", &q17_launcher, "TPC-H Q17 fused hash/aggregate CUDA launcher");
}
'''


def _load_cuda() -> Any:
    """Compile the hand-written CUDA extension from the in-module .cu string.

    The device code is written to a temporary on-disk file because
    torch.utils.cpp_extension.load requires real source paths. The compiled CUDA
    kernels perform the hot numeric path for Q17 on the GPU.
    """
    tmpdir = tempfile.mkdtemp(prefix="q17_cuda_")
    cu_path = os.path.join(tmpdir, "_q17_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q17_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )


_EXT = _load_cuda()


def _launch_q17(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """Launch the CUDA Q17 core.

    Runs entirely on GPU after tensor preparation: builds a compact hash table
    for the filtered part keys, scans lineitem once to compute per-part quantity
    sums/counts, then scans once more to reduce qualifying extended prices. This
    avoids Torch searchsorted masks, scatter temporaries, and multiple eager
    tensor materializations.
    """
    return _EXT.q17_launcher(l_partkey, l_quantity, l_extendedprice, part_keys)


def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC-H Q17: sum(l_extendedprice)/7.0 where l_quantity < 0.2 * avg(l_quantity) over parts
    matching brand/container filters, grouped by partkey.

    CUDA hot path: a hand-written extension builds a GPU hash map for the
    filtered part keys and performs two coalesced lineitem scans with fp64
    accumulators. It is faster than the reference Torch plan because it removes
    sort/searchsorted, boolean compaction, scatter temporaries, and Python/eager
    launch overhead from the numeric core.
    """
    return _launch_q17(l_partkey, l_quantity, l_extendedprice, part_keys)


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                            FROM lineitem
                            WHERE l_partkey = p_partkey);
    Description: TPC-H Q17. Computes the small-quantity order revenue average for parts of a
        given brand and container. Correlated subquery is evaluated as a grouped average over
        all lineitems for each filtered partkey.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table first
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
