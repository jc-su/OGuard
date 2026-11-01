import os
import tempfile

import torch
import cudf
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#define BLOCK_SIZE 1024

__global__ void sorted_count_segments_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    int64_t n,
    double threshold,
    int64_t* __restrict__ unique_counts,
    int64_t* __restrict__ big_counts,
    int32_t* __restrict__ violation
) {
    __shared__ int u_count[BLOCK_SIZE];
    __shared__ int b_count[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int is_unique = 0;
    int is_big = 0;

    if (idx < n) {
        int64_t key = keys[idx];
        if (idx > 0) {
            int64_t prev = keys[idx - 1];
            if (key < prev) {
                atomicExch(violation, 1);
            }
        }

        bool start = (idx == 0) || (key != keys[idx - 1]);
        if (start) {
            is_unique = 1;
            double s = 0.0;
            int64_t j = idx;
            while (j < n && keys[j] == key) {
                s += qty[j];
                ++j;
            }
            is_big = (s > threshold) ? 1 : 0;
        }
    }

    u_count[tid] = is_unique;
    b_count[tid] = is_big;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            u_count[tid] += u_count[tid + offset];
            b_count[tid] += b_count[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unique_counts[blockIdx.x] = static_cast<int64_t>(u_count[0]);
        big_counts[blockIdx.x] = static_cast<int64_t>(b_count[0]);
    }
}

__global__ void scan_counts_kernel(
    const int64_t* __restrict__ in,
    int64_t* __restrict__ out,
    int64_t* __restrict__ block_sums,
    int64_t n
) {
    __shared__ int64_t temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    int64_t x = (gid < n) ? in[gid] : 0;
    temp[tid] = x;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int64_t add = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += add;
        __syncthreads();
    }

    if (gid < n) {
        out[gid] = temp[tid] - x;
    }
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = temp[tid];
    }
}

__global__ void scan_small_kernel(
    const int64_t* __restrict__ in,
    int64_t* __restrict__ out,
    int64_t n
) {
    __shared__ int64_t temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t x = (tid < n) ? in[tid] : 0;
    temp[tid] = x;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int64_t add = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += add;
        __syncthreads();
    }

    if (tid < n) {
        out[tid] = temp[tid] - x;
    }
}

__global__ void add_block_offsets_kernel(
    int64_t* __restrict__ offsets,
    const int64_t* __restrict__ block_offsets,
    int64_t n
) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid < n) {
        offsets[gid] += block_offsets[blockIdx.x];
    }
}

__global__ void total_counts_kernel(
    const int64_t* __restrict__ unique_offsets,
    const int64_t* __restrict__ unique_counts,
    const int64_t* __restrict__ big_offsets,
    const int64_t* __restrict__ big_counts,
    int64_t n_blocks,
    int64_t* __restrict__ totals
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t last = n_blocks - 1;
        totals[0] = unique_offsets[last] + unique_counts[last];
        totals[1] = big_offsets[last] + big_counts[last];
    }
}

__global__ void sorted_compact_segments_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    int64_t n,
    double threshold,
    const int64_t* __restrict__ unique_offsets,
    const int64_t* __restrict__ big_offsets,
    int64_t* __restrict__ unique_out,
    int64_t* __restrict__ big_keys_out,
    double* __restrict__ big_sums_out
) {
    __shared__ int u_scan[BLOCK_SIZE];
    __shared__ int b_scan[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int is_unique = 0;
    int is_big = 0;
    int64_t key = 0;
    double s = 0.0;

    if (idx < n) {
        key = keys[idx];
        bool start = (idx == 0) || (key != keys[idx - 1]);
        if (start) {
            is_unique = 1;
            int64_t j = idx;
            while (j < n && keys[j] == key) {
                s += qty[j];
                ++j;
            }
            is_big = (s > threshold) ? 1 : 0;
        }
    }

    u_scan[tid] = is_unique;
    b_scan[tid] = is_big;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int add_u = (tid >= offset) ? u_scan[tid - offset] : 0;
        int add_b = (tid >= offset) ? b_scan[tid - offset] : 0;
        __syncthreads();
        u_scan[tid] += add_u;
        b_scan[tid] += add_b;
        __syncthreads();
    }

    if (is_unique) {
        int64_t pos = unique_offsets[blockIdx.x] + static_cast<int64_t>(u_scan[tid] - 1);
        unique_out[pos] = key;
    }
    if (is_big) {
        int64_t pos = big_offsets[blockIdx.x] + static_cast<int64_t>(b_scan[tid] - 1);
        big_keys_out[pos] = key;
        big_sums_out[pos] = s;
    }
}

__global__ void max_key_kernel(
    const int64_t* __restrict__ keys,
    int64_t n,
    unsigned long long* __restrict__ max_out
) {
    __shared__ unsigned long long smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    unsigned long long v = 0ULL;

    if (idx < n) {
        int64_t k = keys[idx];
        v = (k > 0) ? static_cast<unsigned long long>(k) : 0ULL;
    }

    smem[tid] = v;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            unsigned long long other = smem[tid + offset];
            if (other > smem[tid]) {
                smem[tid] = other;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_out, smem[0]);
    }
}

__global__ void dense_accumulate_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    int64_t n,
    double* __restrict__ sums,
    int32_t* __restrict__ flags,
    int64_t key_range
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        if (key >= 0 && key < key_range) {
            atomicAdd(&sums[key], qty[idx]);
            atomicExch(&flags[key], 1);
        }
    }
}

__global__ void dense_count_blocks_kernel(
    const int32_t* __restrict__ flags,
    const double* __restrict__ sums,
    int64_t key_range,
    double threshold,
    int64_t* __restrict__ unique_counts,
    int64_t* __restrict__ big_counts
) {
    __shared__ int u_count[BLOCK_SIZE];
    __shared__ int b_count[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t key = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int is_unique = 0;
    int is_big = 0;
    if (key < key_range && flags[key] != 0) {
        is_unique = 1;
        is_big = (sums[key] > threshold) ? 1 : 0;
    }

    u_count[tid] = is_unique;
    b_count[tid] = is_big;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            u_count[tid] += u_count[tid + offset];
            b_count[tid] += b_count[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unique_counts[blockIdx.x] = static_cast<int64_t>(u_count[0]);
        big_counts[blockIdx.x] = static_cast<int64_t>(b_count[0]);
    }
}

__global__ void dense_compact_kernel(
    const int32_t* __restrict__ flags,
    const double* __restrict__ sums,
    int64_t key_range,
    double threshold,
    const int64_t* __restrict__ unique_offsets,
    const int64_t* __restrict__ big_offsets,
    int64_t* __restrict__ unique_out,
    int64_t* __restrict__ big_keys_out,
    double* __restrict__ big_sums_out
) {
    __shared__ int u_scan[BLOCK_SIZE];
    __shared__ int b_scan[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t key = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int is_unique = 0;
    int is_big = 0;
    double s = 0.0;

    if (key < key_range && flags[key] != 0) {
        s = sums[key];
        is_unique = 1;
        is_big = (s > threshold) ? 1 : 0;
    }

    u_scan[tid] = is_unique;
    b_scan[tid] = is_big;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int add_u = (tid >= offset) ? u_scan[tid - offset] : 0;
        int add_b = (tid >= offset) ? b_scan[tid - offset] : 0;
        __syncthreads();
        u_scan[tid] += add_u;
        b_scan[tid] += add_b;
        __syncthreads();
    }

    if (is_unique) {
        int64_t pos = unique_offsets[blockIdx.x] + static_cast<int64_t>(u_scan[tid] - 1);
        unique_out[pos] = key;
    }
    if (is_big) {
        int64_t pos = big_offsets[blockIdx.x] + static_cast<int64_t>(b_scan[tid] - 1);
        big_keys_out[pos] = key;
        big_sums_out[pos] = s;
    }
}

static void exclusive_scan_counts(
    torch::Tensor counts,
    torch::Tensor offsets,
    int64_t n,
    cudaStream_t stream
) {
    int64_t scan_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto block_sums = torch::empty({scan_blocks}, counts.options());
    auto block_offsets = torch::empty({scan_blocks}, counts.options());

    scan_counts_kernel<<<static_cast<int>(scan_blocks), BLOCK_SIZE, 0, stream>>>(
        counts.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        block_sums.data_ptr<int64_t>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scan_small_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        block_sums.data_ptr<int64_t>(),
        block_offsets.data_ptr<int64_t>(),
        scan_blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    add_block_offsets_kernel<<<static_cast<int>(scan_blocks), BLOCK_SIZE, 0, stream>>>(
        offsets.data_ptr<int64_t>(),
        block_offsets.data_ptr<int64_t>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static std::vector<torch::Tensor> dense_groupby_launcher(
    torch::Tensor keys,
    torch::Tensor qty,
    double threshold,
    cudaStream_t stream
) {
    int64_t n = keys.numel();
    auto key_opts = keys.options();
    auto qty_opts = qty.options();
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(keys.device());

    auto max_tensor = torch::zeros({1}, key_opts);
    int line_blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    max_key_kernel<<<line_blocks, BLOCK_SIZE, 0, stream>>>(
        keys.data_ptr<int64_t>(),
        n,
        reinterpret_cast<unsigned long long*>(max_tensor.data_ptr<int64_t>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t host_max = 0;
    cudaMemcpyAsync(&host_max, max_tensor.data_ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t key_range = host_max + 1;
    auto sums = torch::zeros({key_range}, qty_opts);
    auto flags = torch::zeros({key_range}, i32_opts);

    dense_accumulate_kernel<<<line_blocks, BLOCK_SIZE, 0, stream>>>(
        keys.data_ptr<int64_t>(),
        qty.data_ptr<double>(),
        n,
        sums.data_ptr<double>(),
        flags.data_ptr<int32_t>(),
        key_range
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t key_blocks_i64 = (key_range + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int key_blocks = static_cast<int>(key_blocks_i64);

    auto unique_counts = torch::empty({key_blocks_i64}, key_opts);
    auto big_counts = torch::empty({key_blocks_i64}, key_opts);
    auto unique_offsets = torch::empty({key_blocks_i64}, key_opts);
    auto big_offsets = torch::empty({key_blocks_i64}, key_opts);

    dense_count_blocks_kernel<<<key_blocks, BLOCK_SIZE, 0, stream>>>(
        flags.data_ptr<int32_t>(),
        sums.data_ptr<double>(),
        key_range,
        threshold,
        unique_counts.data_ptr<int64_t>(),
        big_counts.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    exclusive_scan_counts(unique_counts, unique_offsets, key_blocks_i64, stream);
    exclusive_scan_counts(big_counts, big_offsets, key_blocks_i64, stream);

    auto unique_full = torch::empty({n}, key_opts);
    auto big_keys_full = torch::empty({n}, key_opts);
    auto big_sums_full = torch::empty({n}, qty_opts);

    dense_compact_kernel<<<key_blocks, BLOCK_SIZE, 0, stream>>>(
        flags.data_ptr<int32_t>(),
        sums.data_ptr<double>(),
        key_range,
        threshold,
        unique_offsets.data_ptr<int64_t>(),
        big_offsets.data_ptr<int64_t>(),
        unique_full.data_ptr<int64_t>(),
        big_keys_full.data_ptr<int64_t>(),
        big_sums_full.data_ptr<double>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto totals = torch::empty({2}, key_opts);
    total_counts_kernel<<<1, 1, 0, stream>>>(
        unique_offsets.data_ptr<int64_t>(),
        unique_counts.data_ptr<int64_t>(),
        big_offsets.data_ptr<int64_t>(),
        big_counts.data_ptr<int64_t>(),
        key_blocks_i64,
        totals.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t host_totals[2] = {0, 0};
    cudaMemcpyAsync(host_totals, totals.data_ptr<int64_t>(), 2 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    auto unique_out = unique_full.slice(0, 0, host_totals[0]).contiguous();
    auto big_keys_out = big_keys_full.slice(0, 0, host_totals[1]).contiguous();
    auto big_sums_out = big_sums_full.slice(0, 0, host_totals[1]).contiguous();

    return {big_keys_out, big_sums_out, unique_out};
}

std::vector<torch::Tensor> q18_launcher(
    torch::Tensor l_orderkey,
    torch::Tensor l_quantity,
    double threshold
) {
    TORCH_CHECK(l_orderkey.scalar_type() == torch::kInt64, "l_orderkey must be int64");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_orderkey.dim() == 1, "l_orderkey must be 1D");
    TORCH_CHECK(l_quantity.dim() == 1, "l_quantity must be 1D");
    TORCH_CHECK(l_orderkey.numel() == l_quantity.numel(), "input sizes must match");

    if (!l_orderkey.is_cuda()) {
        l_orderkey = l_orderkey.to(torch::kCUDA);
    }
    if (!l_quantity.is_cuda()) {
        l_quantity = l_quantity.to(torch::kCUDA);
    }

    l_orderkey = l_orderkey.contiguous();
    l_quantity = l_quantity.contiguous();

    int64_t n = l_orderkey.numel();
    if (n == 0) {
        auto empty_keys = torch::empty({0}, l_orderkey.options());
        auto empty_sums = torch::empty({0}, l_quantity.options());
        return {empty_keys, empty_sums, empty_keys};
    }

    at::cuda::CUDAStream stream_obj = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = stream_obj.stream();

    auto key_opts = l_orderkey.options();
    auto qty_opts = l_quantity.options();
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(l_orderkey.device());

    int64_t line_blocks_i64 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int line_blocks = static_cast<int>(line_blocks_i64);

    auto unique_counts = torch::empty({line_blocks_i64}, key_opts);
    auto big_counts = torch::empty({line_blocks_i64}, key_opts);
    auto violation = torch::zeros({1}, i32_opts);

    sorted_count_segments_kernel<<<line_blocks, BLOCK_SIZE, 0, stream>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        n,
        threshold,
        unique_counts.data_ptr<int64_t>(),
        big_counts.data_ptr<int64_t>(),
        violation.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int32_t host_violation = 0;
    cudaMemcpyAsync(&host_violation, violation.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (host_violation != 0) {
        return dense_groupby_launcher(l_orderkey, l_quantity, threshold, stream);
    }

    auto unique_offsets = torch::empty({line_blocks_i64}, key_opts);
    auto big_offsets = torch::empty({line_blocks_i64}, key_opts);

    exclusive_scan_counts(unique_counts, unique_offsets, line_blocks_i64, stream);
    exclusive_scan_counts(big_counts, big_offsets, line_blocks_i64, stream);

    auto unique_full = torch::empty({n}, key_opts);
    auto big_keys_full = torch::empty({n}, key_opts);
    auto big_sums_full = torch::empty({n}, qty_opts);

    sorted_compact_segments_kernel<<<line_blocks, BLOCK_SIZE, 0, stream>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        n,
        threshold,
        unique_offsets.data_ptr<int64_t>(),
        big_offsets.data_ptr<int64_t>(),
        unique_full.data_ptr<int64_t>(),
        big_keys_full.data_ptr<int64_t>(),
        big_sums_full.data_ptr<double>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto totals = torch::empty({2}, key_opts);
    total_counts_kernel<<<1, 1, 0, stream>>>(
        unique_offsets.data_ptr<int64_t>(),
        unique_counts.data_ptr<int64_t>(),
        big_offsets.data_ptr<int64_t>(),
        big_counts.data_ptr<int64_t>(),
        line_blocks_i64,
        totals.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t host_totals[2] = {0, 0};
    cudaMemcpyAsync(host_totals, totals.data_ptr<int64_t>(), 2 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    auto unique_out = unique_full.slice(0, 0, host_totals[0]).contiguous();
    auto big_keys_out = big_keys_full.slice(0, 0, host_totals[1]).contiguous();
    auto big_sums_out = big_sums_full.slice(0, 0, host_totals[1]).contiguous();

    return {big_keys_out, big_sums_out, unique_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q18_launcher", &q18_launcher, "TPC-H Q18 lineitem order quantity grouping (CUDA)");
}
'''


def _load_cuda() -> object:
    """Compile the CUDA extension from the in-module .cu source on disk.

    CUDA/C++ owns the hot path kernels. Python only writes the source string to a
    temporary file and asks PyTorch's extension loader to build it for H100
    (sm_90). Keeping all grouping kernels in one extension avoids Python-level
    eager tensor operators in `_query_core`.
    """
    tmpdir = tempfile.mkdtemp(prefix="q18_cuda_")
    cu_path = os.path.join(tmpdir, "_q18_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q18_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _query_core(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    qty_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes per-order sum of l_quantity, then filters those orders whose
    summed quantity exceeds the threshold. Returns the unique big order keys,
    along with (for the eventual join) sorted unique orderkeys + sums.

    Runs on GPU in the CUDA extension. The fast path exploits TPC-H lineitem's
    order-key clustering: one CUDA pass counts sorted segments, a block-prefix
    compaction emits sorted unique keys and qualifying sums, and no Torch
    `unique` sort or `scatter_add_` temporaries are created. If the input is not
    nondecreasing, a dense CUDA fallback performs atomic accumulation by
    orderkey and sorted compaction, preserving the reference semantics for the
    benchmark's positive order keys.
    """
    out = _EXT.q18_launcher(l_orderkey, l_quantity, float(qty_threshold))
    return out[0], out[1], out[2]


def run_query(tables: dict[str, cudf.DataFrame], qty_threshold: float = 300.0, limit: int = 100) -> cudf.DataFrame:
    """
    SQL:
      SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
      FROM customer, orders, lineitem
      WHERE o_orderkey IN (
          SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300)
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
      GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
      ORDER BY o_totalprice DESC, o_orderdate
      LIMIT 100;
    Description: TPC-H Q18 — finds the top 100 customers with the largest-quantity orders.
    Params:
      qty_threshold (float): HAVING sum(l_quantity) > threshold.
      limit (int): number of rows to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # --- Compute big orderkeys (HAVING sum(l_quantity) > threshold) in the core ---
    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    big_keys_t, big_sums_t, _ = _query_core(l_orderkey_t, l_quantity_t, float(qty_threshold))

    # Build a cuDF DataFrame of qualifying orderkeys + their total quantity
    df_big = cudf.DataFrame({
        "o_orderkey": cudf.Series(big_keys_t),
        "sum_qty": cudf.Series(big_sums_t),
    })

    # Semi-join orders with big orderkeys, then join with customer
    orders_filtered = df_orders.merge(df_big, left_on="o_orderkey", right_on="o_orderkey", how="inner")
    joined = orders_filtered.merge(
        df_customer[["c_custkey", "c_name"]],
        left_on="o_custkey", right_on="c_custkey", how="inner"
    )

    # Order by o_totalprice DESC, o_orderdate ASC
    joined = joined.sort_values(
        by=["o_totalprice", "o_orderdate"], ascending=[False, True]
    ).head(limit)

    result = cudf.DataFrame({
        "c_name": joined["c_name"].reset_index(drop=True),
        "c_custkey": joined["c_custkey"].reset_index(drop=True),
        "o_orderkey": joined["o_orderkey"].reset_index(drop=True),
        "o_orderdate": joined["o_orderdate"].reset_index(drop=True),
        "o_totalprice": joined["o_totalprice"].reset_index(drop=True),
        "sum(l_quantity)": joined["sum_qty"].reset_index(drop=True),
    })
    return result
