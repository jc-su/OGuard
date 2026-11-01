import os
import tempfile
from typing import Any

import torch
import cudf
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#define BLOCK_SIZE 256
#define HIST_SHARED_CAP 1024

__global__ void q13_count_orders_kernel(
    const int64_t* __restrict__ o_custkey,
    int64_t n_orders,
    int32_t* __restrict__ counts,
    int64_t num_customers
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t key = o_custkey[idx];
        if (key >= 0 && key <= num_customers) {
            atomicAdd(counts + key, 1);
        }
    }
}

__global__ void q13_reduce_max_i32_kernel(
    const int32_t* __restrict__ inp,
    int64_t n,
    int32_t* __restrict__ out
) {
    __shared__ int32_t smem[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t v = 0;
    if (idx < n) {
        v = inp[idx];
    }
    smem[threadIdx.x] = v;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int32_t other = smem[threadIdx.x + stride];
            if (other > smem[threadIdx.x]) {
                smem[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = smem[0];
    }
}

__global__ void q13_hist_customer_counts_kernel(
    const int64_t* __restrict__ c_custkey,
    int64_t n_customers,
    const int32_t* __restrict__ counts,
    int64_t* __restrict__ hist,
    int64_t hist_size,
    int64_t num_customers
) {
    __shared__ unsigned long long shist[HIST_SHARED_CAP];

    bool use_shared = hist_size <= HIST_SHARED_CAP;

    if (use_shared) {
        for (int64_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
            shist[i] = 0ULL;
        }
        __syncthreads();
    }

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_customers) {
        int64_t key = c_custkey[idx];
        if (key >= 0 && key <= num_customers) {
            int32_t cnt = counts[key];
            if (cnt >= 0 && static_cast<int64_t>(cnt) < hist_size) {
                if (use_shared) {
                    atomicAdd(&shist[cnt], 1ULL);
                } else {
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(hist + cnt),
                        1ULL
                    );
                }
            }
        }
    }

    if (use_shared) {
        __syncthreads();
        for (int64_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
            unsigned long long v = shist[i];
            if (v != 0ULL) {
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(hist + i),
                    v
                );
            }
        }
    }
}

int32_t q13_device_max_i32(torch::Tensor inp, int64_t n, cudaStream_t stream) {
    if (n <= 0) {
        return 0;
    }

    torch::Tensor current = inp;
    int64_t current_n = n;
    auto opts = torch::TensorOptions().device(inp.device()).dtype(torch::kInt32);

    while (current_n > 1) {
        int64_t blocks64 = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        torch::Tensor next = torch::empty({blocks64}, opts);
        q13_reduce_max_i32_kernel<<<static_cast<int>(blocks64), BLOCK_SIZE, 0, stream>>>(
            current.data_ptr<int32_t>(),
            current_n,
            next.data_ptr<int32_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        current = next;
        current_n = blocks64;
    }

    int32_t hmax = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &hmax,
        current.data_ptr<int32_t>(),
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return hmax;
}

std::vector<torch::Tensor> q13_launcher(
    torch::Tensor c_custkey,
    torch::Tensor o_custkey_filtered,
    int64_t num_customers
) {
    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64, "c_custkey must be int64");
    TORCH_CHECK(o_custkey_filtered.scalar_type() == torch::kInt64, "o_custkey_filtered must be int64");
    TORCH_CHECK(num_customers >= 0, "num_customers must be non-negative");

    if (!c_custkey.is_cuda()) {
        c_custkey = c_custkey.cuda();
    }
    if (!o_custkey_filtered.is_cuda()) {
        o_custkey_filtered = o_custkey_filtered.cuda();
    }

    c10::cuda::CUDAGuard device_guard(c_custkey.device());
    c_custkey = c_custkey.contiguous();
    o_custkey_filtered = o_custkey_filtered.contiguous();

    const int64_t n_customers = c_custkey.numel();
    const int64_t n_orders = o_custkey_filtered.numel();

    auto int32_opts = torch::TensorOptions().device(c_custkey.device()).dtype(torch::kInt32);
    auto int64_opts = torch::TensorOptions().device(c_custkey.device()).dtype(torch::kInt64);

    if (n_customers <= 0) {
        return {
            torch::empty({0}, int64_opts),
            torch::empty({0}, int64_opts)
        };
    }

    torch::Tensor counts = torch::zeros({num_customers + 1}, int32_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n_orders > 0) {
        const int blocks_orders = static_cast<int>((n_orders + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q13_count_orders_kernel<<<blocks_orders, BLOCK_SIZE, 0, stream>>>(
            o_custkey_filtered.data_ptr<int64_t>(),
            n_orders,
            counts.data_ptr<int32_t>(),
            num_customers
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int32_t max_count_i32 = q13_device_max_i32(counts, num_customers + 1, stream);
    if (max_count_i32 < 0) {
        max_count_i32 = 0;
    }

    const int64_t hist_size = static_cast<int64_t>(max_count_i32) + 1;
    torch::Tensor hist = torch::zeros({hist_size}, int64_opts);

    const int blocks_customers = static_cast<int>((n_customers + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q13_hist_customer_counts_kernel<<<blocks_customers, BLOCK_SIZE, 0, stream>>>(
        c_custkey.data_ptr<int64_t>(),
        n_customers,
        counts.data_ptr<int32_t>(),
        hist.data_ptr<int64_t>(),
        hist_size,
        num_customers
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> h_hist(hist_size);
    C10_CUDA_CHECK(cudaMemcpyAsync(
        h_hist.data(),
        hist.data_ptr<int64_t>(),
        hist_size * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::pair<int64_t, int64_t>> rows;
    rows.reserve(hist_size);
    for (int64_t c = 0; c < hist_size; ++c) {
        int64_t d = h_hist[c];
        if (d != 0) {
            rows.emplace_back(c, d);
        }
    }

    std::sort(
        rows.begin(),
        rows.end(),
        [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
            if (a.second != b.second) {
                return a.second > b.second;
            }
            return a.first > b.first;
        }
    );

    const int64_t k = static_cast<int64_t>(rows.size());
    torch::Tensor final_counts = torch::empty({k}, int64_opts);
    torch::Tensor final_dist = torch::empty({k}, int64_opts);

    if (k > 0) {
        std::vector<int64_t> h_counts(k);
        std::vector<int64_t> h_dist(k);
        for (int64_t i = 0; i < k; ++i) {
            h_counts[i] = rows[i].first;
            h_dist[i] = rows[i].second;
        }

        C10_CUDA_CHECK(cudaMemcpyAsync(
            final_counts.data_ptr<int64_t>(),
            h_counts.data(),
            k * sizeof(int64_t),
            cudaMemcpyHostToDevice,
            stream
        ));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            final_dist.data_ptr<int64_t>(),
            h_dist.data(),
            k * sizeof(int64_t),
            cudaMemcpyHostToDevice,
            stream
        ));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return {final_counts, final_dist};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q13_launcher", &q13_launcher, "TPC-H Q13 customer distribution core (CUDA)");
}
'''


def _load_cuda() -> Any:
    tmpdir = tempfile.mkdtemp(prefix="q13_cuda_")
    cu_path = os.path.join(tmpdir, "_q13_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q13_cuda_ext_v2",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )


_EXT = _load_cuda()


def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU/CUDA core for TPC-H Q13 after cuDF has filtered orders by comment.

    CUDA runs the numeric hot path: filtered order customer keys are counted with
    one coalesced scan plus atomics, the maximum per-customer count is reduced on
    GPU, then only the small required count histogram is allocated and filled.
    This avoids PyTorch scatter/gather/unique temporaries and avoids scanning a
    huge order-sized histogram, which is faster for the memory-bound Q13 core.
    """
    result = _EXT.q13_launcher(c_custkey, o_custkey_filtered, int(num_customers))
    return result[0], result[1]


def run_query(tables: dict[str, cudf.DataFrame], comment_pattern: str = "%special%requests%") -> cudf.DataFrame:
    """
    SQL:
        SELECT c_count, count(*) AS custdist
        FROM (
            SELECT c_custkey, count(o_orderkey)
            FROM customer LEFT OUTER JOIN orders
              ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'
            GROUP BY c_custkey
        ) AS c_orders (c_custkey, c_count)
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;
    Description: TPC-H Q13 - Customer distribution based on number of non-special-request orders.
    Params: comment_pattern (str): the LIKE pattern to exclude from orders via NOT LIKE.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Filter orders: o_comment NOT LIKE '%special%requests%'
    # cuDF supports .str.contains with regex. Convert SQL LIKE pattern to regex: % -> .*
    # For the specific pattern, '%special%requests%' -> '.*special.*requests.*' (equivalent to contains 'special.*requests')
    mask = ~df_orders["o_comment"].str.contains("special.*requests", regex=True)
    filtered_orders = df_orders[mask]

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    o_custkey_t = torch.as_tensor(filtered_orders["o_custkey"].to_cupy(), dtype=torch.int64)

    # Use max custkey as the bucket size upper bound
    num_customers = int(df_customer["c_custkey"].max())

    c_count_t, custdist_t = _query_core(c_custkey_t, o_custkey_t, num_customers)

    return cudf.DataFrame({
        "c_count": c_count_t,
        "custdist": custdist_t,
    })
