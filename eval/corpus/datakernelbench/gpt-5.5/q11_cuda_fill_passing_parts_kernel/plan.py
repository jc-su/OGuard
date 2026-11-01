import os
import tempfile
from typing import Tuple

import cudf
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <vector>

#define BLOCK_SIZE 1024

__global__ void mark_suppliers_kernel(
    const int64_t* __restrict__ s_suppkey,
    const int64_t* __restrict__ s_nationkey,
    const int64_t* __restrict__ nation_keys,
    int64_t n_suppliers,
    int64_t n_nation_keys,
    uint8_t* __restrict__ supplier_keep,
    int64_t supplier_keep_len
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_suppliers) {
        return;
    }

    int64_t nk = s_nationkey[idx];
    bool keep = false;
    #pragma unroll 1
    for (int64_t j = 0; j < n_nation_keys; ++j) {
        if (nk == nation_keys[j]) {
            keep = true;
            break;
        }
    }

    if (keep) {
        int64_t sk = s_suppkey[idx];
        if (sk >= 0 && sk < supplier_keep_len) {
            supplier_keep[sk] = static_cast<uint8_t>(1);
        }
    }
}

__global__ void aggregate_partsupp_kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    const double* __restrict__ ps_supplycost,
    const int64_t* __restrict__ ps_availqty,
    const uint8_t* __restrict__ supplier_keep,
    int64_t supplier_keep_len,
    double* __restrict__ part_sums,
    int64_t part_sums_len,
    double* __restrict__ total_sum,
    int64_t n_partsupp
) {
    __shared__ double block_total[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double local = 0.0;

    if (idx < n_partsupp) {
        int64_t sk = ps_suppkey[idx];
        if (sk >= 0 && sk < supplier_keep_len && supplier_keep[sk] != 0) {
            int64_t pk = ps_partkey[idx];
            if (pk >= 0 && pk < part_sums_len) {
                double v = ps_supplycost[idx] * static_cast<double>(ps_availqty[idx]);
                atomicAdd(&part_sums[pk], v);
                local = v;
            }
        }
    }

    block_total[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_total[threadIdx.x] += block_total[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(total_sum, block_total[0]);
    }
}

__global__ void count_passing_parts_kernel(
    const double* __restrict__ part_sums,
    const double* __restrict__ total_sum,
    double threshold_fraction,
    int64_t part_sums_len,
    int64_t* __restrict__ out_count
) {
    __shared__ int block_count[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double threshold = total_sum[0] * threshold_fraction;
    int local = 0;

    if (idx < part_sums_len) {
        double v = part_sums[idx];
        local = (v > threshold) ? 1 : 0;
    }

    block_count[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_count[threadIdx.x] += block_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && block_count[0] != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(out_count),
                  static_cast<unsigned long long>(block_count[0]));
    }
}

__global__ void fill_passing_parts_kernel(
    const double* __restrict__ part_sums,
    const double* __restrict__ total_sum,
    double threshold_fraction,
    int64_t part_sums_len,
    int64_t* __restrict__ out_keys,
    double* __restrict__ out_values,
    int64_t* __restrict__ out_pos
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= part_sums_len) {
        return;
    }

    double v = part_sums[idx];
    double threshold = total_sum[0] * threshold_fraction;
    if (v > threshold) {
        unsigned long long pos = atomicAdd(
            reinterpret_cast<unsigned long long*>(out_pos),
            static_cast<unsigned long long>(1)
        );
        out_keys[static_cast<int64_t>(pos)] = idx;
        out_values[static_cast<int64_t>(pos)] = v;
    }
}

static torch::Tensor ensure_on_device_contiguous(torch::Tensor t, const torch::Device& device) {
    if (!t.is_cuda() || t.device() != device) {
        t = t.to(device);
    }
    return t.contiguous();
}

std::vector<torch::Tensor> q11_launcher(
    torch::Tensor ps_partkey,
    torch::Tensor ps_suppkey,
    torch::Tensor ps_supplycost,
    torch::Tensor ps_availqty,
    torch::Tensor s_suppkey,
    torch::Tensor s_nationkey,
    torch::Tensor nation_keys,
    int64_t max_partkey,
    int64_t max_suppkey,
    double threshold_fraction
) {
    if (!ps_partkey.is_cuda()) {
        ps_partkey = ps_partkey.to(torch::kCUDA);
    }
    ps_partkey = ps_partkey.contiguous();
    const torch::Device device = ps_partkey.device();
    c10::cuda::CUDAGuard device_guard(device);

    ps_suppkey = ensure_on_device_contiguous(ps_suppkey, device);
    ps_supplycost = ensure_on_device_contiguous(ps_supplycost, device);
    ps_availqty = ensure_on_device_contiguous(ps_availqty, device);
    s_suppkey = ensure_on_device_contiguous(s_suppkey, device);
    s_nationkey = ensure_on_device_contiguous(s_nationkey, device);
    nation_keys = ensure_on_device_contiguous(nation_keys, device);

    TORCH_CHECK(ps_partkey.scalar_type() == torch::kInt64, "ps_partkey must be int64");
    TORCH_CHECK(ps_suppkey.scalar_type() == torch::kInt64, "ps_suppkey must be int64");
    TORCH_CHECK(ps_supplycost.scalar_type() == torch::kFloat64, "ps_supplycost must be float64");
    TORCH_CHECK(ps_availqty.scalar_type() == torch::kInt64, "ps_availqty must be int64");
    TORCH_CHECK(s_suppkey.scalar_type() == torch::kInt64, "s_suppkey must be int64");
    TORCH_CHECK(s_nationkey.scalar_type() == torch::kInt64, "s_nationkey must be int64");
    TORCH_CHECK(nation_keys.scalar_type() == torch::kInt64, "nation_keys must be int64");
    TORCH_CHECK(max_partkey >= 0, "max_partkey must be non-negative");
    TORCH_CHECK(max_suppkey >= 0, "max_suppkey must be non-negative");

    const int64_t n_partsupp = ps_partkey.numel();
    const int64_t n_suppliers = s_suppkey.numel();
    const int64_t n_nation_keys = nation_keys.numel();

    TORCH_CHECK(ps_suppkey.numel() == n_partsupp, "ps_suppkey length mismatch");
    TORCH_CHECK(ps_supplycost.numel() == n_partsupp, "ps_supplycost length mismatch");
    TORCH_CHECK(ps_availqty.numel() == n_partsupp, "ps_availqty length mismatch");
    TORCH_CHECK(s_nationkey.numel() == n_suppliers, "s_nationkey length mismatch");

    auto int_opts = ps_partkey.options().dtype(torch::kInt64);
    auto double_opts = ps_supplycost.options().dtype(torch::kFloat64);
    auto byte_opts = ps_partkey.options().dtype(torch::kUInt8);

    const int64_t supplier_keep_len = max_suppkey + 1;
    const int64_t part_sums_len = max_partkey + 1;

    torch::Tensor supplier_keep = torch::zeros({supplier_keep_len}, byte_opts);
    torch::Tensor part_sums = torch::zeros({part_sums_len}, double_opts);
    torch::Tensor total_sum = torch::zeros({1}, double_opts);
    torch::Tensor count = torch::zeros({1}, int_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n_suppliers > 0 && n_nation_keys > 0) {
        int blocks_sup = static_cast<int>((n_suppliers + BLOCK_SIZE - 1) / BLOCK_SIZE);
        mark_suppliers_kernel<<<blocks_sup, BLOCK_SIZE, 0, stream>>>(
            s_suppkey.data_ptr<int64_t>(),
            s_nationkey.data_ptr<int64_t>(),
            nation_keys.data_ptr<int64_t>(),
            n_suppliers,
            n_nation_keys,
            supplier_keep.data_ptr<uint8_t>(),
            supplier_keep_len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (n_partsupp > 0 && n_nation_keys > 0) {
        int blocks_ps = static_cast<int>((n_partsupp + BLOCK_SIZE - 1) / BLOCK_SIZE);
        aggregate_partsupp_kernel<<<blocks_ps, BLOCK_SIZE, 0, stream>>>(
            ps_partkey.data_ptr<int64_t>(),
            ps_suppkey.data_ptr<int64_t>(),
            ps_supplycost.data_ptr<double>(),
            ps_availqty.data_ptr<int64_t>(),
            supplier_keep.data_ptr<uint8_t>(),
            supplier_keep_len,
            part_sums.data_ptr<double>(),
            part_sums_len,
            total_sum.data_ptr<double>(),
            n_partsupp
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (part_sums_len > 0) {
        int blocks_parts = static_cast<int>((part_sums_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
        count_passing_parts_kernel<<<blocks_parts, BLOCK_SIZE, 0, stream>>>(
            part_sums.data_ptr<double>(),
            total_sum.data_ptr<double>(),
            threshold_fraction,
            part_sums_len,
            count.data_ptr<int64_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int64_t h_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &h_count,
        count.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    torch::Tensor out_keys = torch::empty({h_count}, int_opts);
    torch::Tensor out_values = torch::empty({h_count}, double_opts);

    if (h_count > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(count.data_ptr<int64_t>(), 0, sizeof(int64_t), stream));
        int blocks_parts = static_cast<int>((part_sums_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
        fill_passing_parts_kernel<<<blocks_parts, BLOCK_SIZE, 0, stream>>>(
            part_sums.data_ptr<double>(),
            total_sum.data_ptr<double>(),
            threshold_fraction,
            part_sums_len,
            out_keys.data_ptr<int64_t>(),
            out_values.data_ptr<double>(),
            count.data_ptr<int64_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {out_keys, out_values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q11_launcher", &q11_launcher, "TPC-H Q11 fused supplier filter, dense part aggregate, and HAVING compaction (CUDA)");
}
'''


def _load_cuda() -> object:
    """
    Compile the CUDA extension from the in-module source string.

    The generated .cu file contains the hot GPU kernels.  The Python host only
    prepares cuDF/Torch views and calls the compiled launcher, avoiding Python
    work in the memory-bound scan/aggregate path.
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
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _empty_result() -> cudf.DataFrame:
    """
    Build an empty cuDF result with the exact SELECT column names and dtypes.

    This runs in cuDF/Python only for degenerate inputs such as a missing nation
    literal; it is not on the benchmark hot path.
    """
    return cudf.DataFrame(
        {
            "ps_partkey": cudf.Series([], dtype="int64"),
            "value": cudf.Series([], dtype="float64"),
        }
    )


def _query_core(
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    s_suppkey: torch.Tensor,
    s_nationkey: torch.Tensor,
    nation_keys: torch.Tensor,
    max_partkey: int,
    max_suppkey: int,
    threshold_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the hot path in hand-written CUDA.

    CUDA marks suppliers in the requested nation, scans partsupp once with
    coalesced loads, conditionally projects supplycost*availqty, atomically
    accumulates into a dense ps_partkey array, computes the global threshold,
    and compacts HAVING-passing groups.  This avoids materializing cuDF joins
    and avoids PyTorch unique/inverse/scatter temporaries over the joined rows.
    """
    out = _EXT.q11_launcher(
        ps_partkey,
        ps_suppkey,
        ps_supplycost,
        ps_availqty,
        s_suppkey,
        s_nationkey,
        nation_keys,
        int(max_partkey),
        int(max_suppkey),
        float(threshold_fraction),
    )
    return out[0], out[1]


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "GERMANY",
    threshold_fraction: float = 0.0001,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey
          AND s_nationkey = n_nationkey
          AND n_name = 'GERMANY'
        GROUP BY ps_partkey
        HAVING sum(ps_supplycost * ps_availqty) > (
            SELECT sum(ps_supplycost * ps_availqty) * 0.0001000000
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey
              AND s_nationkey = n_nationkey
              AND n_name = 'GERMANY')
        ORDER BY value DESC;

    Description:
        TPC-H Q11.  cuDF is used only for the tiny string filter on nation and
        for the final ORDER BY of the compact HAVING result.  The bulk work runs
        in fused CUDA: supplier-nation membership is converted to a GPU bitmap,
        partsupp is scanned once, value is projected in registers, per-part
        sums are accumulated in a dense GPU array, the global threshold is
        derived, and passing groups are compacted.  This should be faster than
        the reference because it removes the large cuDF join materialization and
        PyTorch unique/inverse/scatter passes, replacing them with one
        memory-bandwidth-oriented CUDA scan plus a dense post-scan.

    Params:
        tables (dict[str, cudf.DataFrame]): input TPC-H tables containing
            partsupp, supplier, and nation.
        nation_name (str): nation name filter literal.
        threshold_fraction (float): fractional threshold used in HAVING.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    filtered_nation = df_nation[df_nation["n_name"] == nation_name]
    if len(filtered_nation) == 0 or len(df_partsupp) == 0 or len(df_supplier) == 0:
        return _empty_result()

    nation_keys_t = torch.as_tensor(
        filtered_nation["n_nationkey"].to_cupy(), dtype=torch.int64
    )

    ps_partkey_t = torch.as_tensor(df_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(df_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(
        df_partsupp["ps_supplycost"].to_cupy(), dtype=torch.float64
    )
    ps_availqty_t = torch.as_tensor(df_partsupp["ps_availqty"].to_cupy(), dtype=torch.int64)

    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)
    s_nationkey_t = torch.as_tensor(df_supplier["s_nationkey"].to_cupy(), dtype=torch.int64)

    max_partkey = int(df_partsupp["ps_partkey"].max())
    max_suppkey = max(int(df_supplier["s_suppkey"].max()), int(df_partsupp["ps_suppkey"].max()))

    keys_t, values_t = _query_core(
        ps_partkey_t,
        ps_suppkey_t,
        ps_supplycost_t,
        ps_availqty_t,
        s_suppkey_t,
        s_nationkey_t,
        nation_keys_t,
        max_partkey,
        max_suppkey,
        threshold_fraction,
    )

    result = cudf.DataFrame({"ps_partkey": keys_t, "value": values_t})
    if len(result) == 0:
        return result
    return result.sort_values("value", ascending=False).reset_index(drop=True)
