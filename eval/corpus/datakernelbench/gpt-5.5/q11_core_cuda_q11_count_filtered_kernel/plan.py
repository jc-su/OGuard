import os
import tempfile
import types

import torch
import cudf
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 256

__global__ void q11_aggregate_kernel(
    const int64_t* __restrict__ partkey_ptr,
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ availqty_ptr,
    int64_t n_elements,
    double* __restrict__ dense_values_ptr,
    int32_t* __restrict__ present_ptr,
    double* __restrict__ total_ptr
) {
    __shared__ double block_sum[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    double v = 0.0;
    if (idx < n_elements) {
        const int64_t key = partkey_ptr[idx];
        v = supplycost_ptr[idx] * availqty_ptr[idx];

        if (key >= 0) {
            atomicAdd(dense_values_ptr + key, v);
            present_ptr[key] = 1;
        }
    }

    block_sum[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_sum[tid] += block_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total_ptr, block_sum[0]);
    }
}

__global__ void q11_count_filtered_kernel(
    const double* __restrict__ dense_values_ptr,
    const int32_t* __restrict__ present_ptr,
    int64_t dense_len,
    double threshold,
    unsigned long long* __restrict__ count_ptr
) {
    __shared__ unsigned int block_count[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    unsigned int keep = 0;
    if (idx < dense_len) {
        keep = (present_ptr[idx] != 0 && dense_values_ptr[idx] > threshold) ? 1u : 0u;
    }

    block_count[tid] = keep;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_count[tid] += block_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && block_count[0] != 0u) {
        atomicAdd(count_ptr, static_cast<unsigned long long>(block_count[0]));
    }
}

__global__ void q11_compact_filtered_kernel(
    const double* __restrict__ dense_values_ptr,
    const int32_t* __restrict__ present_ptr,
    int64_t dense_len,
    double threshold,
    int64_t* __restrict__ out_keys_ptr,
    double* __restrict__ out_values_ptr,
    unsigned long long* __restrict__ write_ptr
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < dense_len) {
        const double v = dense_values_ptr[idx];
        if (present_ptr[idx] != 0 && v > threshold) {
            const unsigned long long pos = atomicAdd(write_ptr, 1ULL);
            out_keys_ptr[pos] = idx;
            out_values_ptr[pos] = v;
        }
    }
}

std::vector<torch::Tensor> q11_launcher(
    torch::Tensor ps_partkey,
    torch::Tensor ps_supplycost,
    torch::Tensor ps_availqty,
    double threshold_fraction
) {
    TORCH_CHECK(ps_partkey.scalar_type() == torch::kInt64, "ps_partkey must be int64");
    TORCH_CHECK(ps_supplycost.scalar_type() == torch::kFloat64, "ps_supplycost must be float64");
    TORCH_CHECK(ps_availqty.scalar_type() == torch::kFloat64, "ps_availqty must be float64");
    TORCH_CHECK(ps_partkey.numel() == ps_supplycost.numel(), "ps_partkey and ps_supplycost length mismatch");
    TORCH_CHECK(ps_partkey.numel() == ps_availqty.numel(), "ps_partkey and ps_availqty length mismatch");

    if (!ps_partkey.is_cuda()) {
        ps_partkey = ps_partkey.cuda();
    }
    if (!ps_supplycost.is_cuda()) {
        ps_supplycost = ps_supplycost.cuda();
    }
    if (!ps_availqty.is_cuda()) {
        ps_availqty = ps_availqty.cuda();
    }

    TORCH_CHECK(ps_partkey.is_cuda(), "ps_partkey must be CUDA after conversion");
    TORCH_CHECK(ps_supplycost.is_cuda(), "ps_supplycost must be CUDA after conversion");
    TORCH_CHECK(ps_availqty.is_cuda(), "ps_availqty must be CUDA after conversion");

    ps_partkey = ps_partkey.contiguous();
    ps_supplycost = ps_supplycost.contiguous();
    ps_availqty = ps_availqty.contiguous();

    const int64_t n = ps_partkey.numel();
    auto key_options = ps_partkey.options();
    auto value_options = ps_supplycost.options();

    if (n == 0) {
        return {
            torch::empty({0}, key_options),
            torch::empty({0}, value_options)
        };
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    thrust::device_ptr<int64_t> key_begin =
        thrust::device_pointer_cast(ps_partkey.data_ptr<int64_t>());
    const int64_t max_key = thrust::reduce(
        thrust::cuda::par.on(stream),
        key_begin,
        key_begin + n,
        static_cast<int64_t>(0),
        thrust::maximum<int64_t>()
    );

    TORCH_CHECK(max_key >= 0, "ps_partkey contains only negative keys or invalid max key");

    const int64_t dense_len = max_key + 1;
    auto dense_values = torch::zeros({dense_len}, value_options);
    auto present = torch::zeros({dense_len}, key_options.dtype(torch::kInt32));
    auto total = torch::zeros({1}, value_options);

    const int blocks_rows = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q11_aggregate_kernel<<<blocks_rows, BLOCK_SIZE, 0, stream>>>(
        ps_partkey.data_ptr<int64_t>(),
        ps_supplycost.data_ptr<double>(),
        ps_availqty.data_ptr<double>(),
        n,
        dense_values.data_ptr<double>(),
        present.data_ptr<int32_t>(),
        total.data_ptr<double>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const double total_host = total.item<double>();
    const double threshold = total_host * threshold_fraction;

    auto count_tensor = torch::zeros({1}, key_options);
    const int blocks_dense = static_cast<int>((dense_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q11_count_filtered_kernel<<<blocks_dense, BLOCK_SIZE, 0, stream>>>(
        dense_values.data_ptr<double>(),
        present.data_ptr<int32_t>(),
        dense_len,
        threshold,
        reinterpret_cast<unsigned long long*>(count_tensor.data_ptr<int64_t>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t out_count = count_tensor.item<int64_t>();
    auto out_keys = torch::empty({out_count}, key_options);
    auto out_values = torch::empty({out_count}, value_options);

    if (out_count == 0) {
        return {out_keys, out_values};
    }

    auto write_tensor = torch::zeros({1}, key_options);
    q11_compact_filtered_kernel<<<blocks_dense, BLOCK_SIZE, 0, stream>>>(
        dense_values.data_ptr<double>(),
        present.data_ptr<int32_t>(),
        dense_len,
        threshold,
        out_keys.data_ptr<int64_t>(),
        out_values.data_ptr<double>(),
        reinterpret_cast<unsigned long long*>(write_tensor.data_ptr<int64_t>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (out_count > 1) {
        thrust::device_ptr<double> value_sort_begin =
            thrust::device_pointer_cast(out_values.data_ptr<double>());
        thrust::device_ptr<int64_t> key_sort_begin =
            thrust::device_pointer_cast(out_keys.data_ptr<int64_t>());

        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            value_sort_begin,
            value_sort_begin + out_count,
            key_sort_begin,
            thrust::greater<double>()
        );
    }

    return {out_keys, out_values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q11_launcher", &q11_launcher, "TPC-H Q11 grouped value aggregation and descending sort (CUDA)");
}
'''


def _load_cuda() -> types.ModuleType:
    """
    Compile the CUDA extension from the embedded .cu source.

    The host Python module writes the CUDA source to a temporary file because
    torch.utils.cpp_extension.load needs an on-disk translation unit. The hot
    path runs on the GPU: dense partkey aggregation, HAVING count/compact, and
    device-side descending sort.
    """
    tmpdir = tempfile.mkdtemp(prefix="q11_cuda_")
    cu_path = os.path.join(tmpdir, "_q11_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q11_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90", "--expt-relaxed-constexpr"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _launch_q11(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the Q11 numeric core on the GPU through the compiled CUDA extension.

    The computation is mostly a memory-bound scan plus grouped accumulation.
    The CUDA path avoids PyTorch unique/inverse/scatter temporaries by
    aggregating directly into a dense GPU array keyed by ps_partkey, then
    compacts only qualifying groups before sorting.
    """
    result = _EXT.q11_launcher(
        ps_partkey,
        ps_supplycost,
        ps_availqty,
        float(threshold_fraction),
    )
    return result[0], result[1]


def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA GPU tensor core for TPC-H Q11 after cuDF joins.

    This replaces the eager PyTorch unique + scatter_add hot path with CUDA
    aggregation over joined rows, followed by GPU HAVING compaction and
    device-side ordering. It should be faster by reducing large temporary
    tensors and full passes over the joined row set.
    """
    return _launch_q11(
        ps_partkey,
        ps_supplycost,
        ps_availqty,
        threshold_fraction,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "GERMANY",
    threshold_fraction: float = 0.0001,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        GROUP BY ps_partkey
        HAVING sum(ps_supplycost * ps_availqty) > (
            SELECT sum(ps_supplycost * ps_availqty) * 0.0001
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        )
        ORDER BY value DESC;
    Description: TPC-H Q11 - Identifies parts contributing significantly to the total
                 supply cost value for suppliers in a given nation.
    Params:
        nation_name (str): nation name filter literal.
        threshold_fraction (float): fractional threshold used in HAVING clause.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    # Filter nation
    filtered_nation = df_nation[df_nation["n_name"] == nation_name]

    # Join supplier <-> nation
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )

    # Join partsupp <-> (supplier <-> nation)
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")

    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)

    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    return cudf.DataFrame({
        "ps_partkey": keys_t,
        "value": values_t,
    })
