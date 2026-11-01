import os
import tempfile
from typing import Any

import cudf
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void q18_check_sorted_max_kernel(
    const int64_t* __restrict__ keys,
    int64_t* __restrict__ max_out,
    int32_t* __restrict__ sorted_flag,
    int64_t n
) {
    __shared__ int64_t smax[BLOCK_SIZE];
    __shared__ int32_t ssorted[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t local_max = 0;
    int32_t local_sorted = 1;

    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
        int64_t k = keys[idx];
        if (k > local_max) local_max = k;
        if (idx > 0) {
            int64_t prev = keys[idx - 1];
            if (k < prev) local_sorted = 0;
        }
    }

    smax[tid] = local_max;
    ssorted[tid] = local_sorted;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
            ssorted[tid] = ssorted[tid] & ssorted[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((unsigned long long*)max_out, (unsigned long long)smax[0]);
        if (ssorted[0] == 0) {
            atomicExch(sorted_flag, 0);
        }
    }
}

__global__ void q18_count_big_sorted_runs_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    double threshold,
    unsigned long long* __restrict__ count_out,
    int64_t n
) {
    __shared__ unsigned int scount[BLOCK_SIZE];

    int tid = threadIdx.x;
    unsigned int local_count = 0;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
        int64_t k = keys[idx];
        bool is_start = (idx == 0) || (keys[idx - 1] != k);
        if (is_start) {
            double sum = qty[idx];
            int64_t j = idx + 1;
            while (j < n && keys[j] == k) {
                sum += qty[j];
                ++j;
            }
            if (sum > threshold) {
                ++local_count;
            }
        }
    }

    scount[tid] = local_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && scount[0] != 0) {
        atomicAdd(count_out, (unsigned long long)scount[0]);
    }
}

__global__ void q18_fill_big_sorted_runs_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    double threshold,
    int64_t* __restrict__ out_keys,
    double* __restrict__ out_sums,
    unsigned long long* __restrict__ pos_out,
    int64_t n
) {
    int tid = threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
        int64_t k = keys[idx];
        bool is_start = (idx == 0) || (keys[idx - 1] != k);
        if (is_start) {
            double sum = qty[idx];
            int64_t j = idx + 1;
            while (j < n && keys[j] == k) {
                sum += qty[j];
                ++j;
            }
            if (sum > threshold) {
                unsigned long long pos = atomicAdd(pos_out, 1ULL);
                out_keys[pos] = k;
                out_sums[pos] = sum;
            }
        }
    }
}

__global__ void q18_atomic_aggregate_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    double* __restrict__ sums,
    int64_t n,
    int64_t sums_n
) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        int64_t k = keys[idx];
        if (k >= 0 && k < sums_n) {
            atomicAdd(&sums[k], qty[idx]);
        }
    }
}

__global__ void q18_count_big_dense_kernel(
    const double* __restrict__ sums,
    double threshold,
    unsigned long long* __restrict__ count_out,
    int64_t n
) {
    __shared__ unsigned int scount[BLOCK_SIZE];

    int tid = threadIdx.x;
    unsigned int local_count = 0;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
        if (sums[idx] > threshold) {
            ++local_count;
        }
    }

    scount[tid] = local_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && scount[0] != 0) {
        atomicAdd(count_out, (unsigned long long)scount[0]);
    }
}

__global__ void q18_fill_big_dense_kernel(
    const double* __restrict__ sums,
    double threshold,
    int64_t* __restrict__ out_keys,
    double* __restrict__ out_sums,
    unsigned long long* __restrict__ pos_out,
    int64_t n
) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        double s = sums[idx];
        if (s > threshold) {
            unsigned long long pos = atomicAdd(pos_out, 1ULL);
            out_keys[pos] = idx;
            out_sums[pos] = s;
        }
    }
}

static inline int pick_blocks(int64_t n) {
    int64_t blocks64 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks64 < 1) blocks64 = 1;
    if (blocks64 > 131072) blocks64 = 131072;
    return (int)blocks64;
}

std::vector<torch::Tensor> q18_aggregate_launcher(
    torch::Tensor orderkeys,
    torch::Tensor quantity,
    double threshold
) {
    TORCH_CHECK(orderkeys.scalar_type() == torch::kInt64, "l_orderkey must be int64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(orderkeys.numel() == quantity.numel(), "l_orderkey and l_quantity length mismatch");

    if (!orderkeys.is_cuda()) orderkeys = orderkeys.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();

    orderkeys = orderkeys.contiguous();
    quantity = quantity.contiguous();

    int64_t n = orderkeys.numel();
    auto key_opts = orderkeys.options().dtype(torch::kInt64);
    auto qty_opts = quantity.options().dtype(torch::kFloat64);

    if (n == 0) {
        return {
            torch::empty({0}, key_opts),
            torch::empty({0}, qty_opts)
        };
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto max_tensor = torch::zeros({1}, key_opts);
    auto sorted_flag_tensor = torch::ones({1}, orderkeys.options().dtype(torch::kInt32));

    int blocks_n = pick_blocks(n);
    q18_check_sorted_max_kernel<<<blocks_n, BLOCK_SIZE, 0, stream>>>(
        orderkeys.data_ptr<int64_t>(),
        max_tensor.data_ptr<int64_t>(),
        sorted_flag_tensor.data_ptr<int32_t>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int32_t is_sorted = sorted_flag_tensor.item<int32_t>();

    if (is_sorted != 0) {
        auto count_tensor = torch::zeros({1}, orderkeys.options().dtype(torch::kInt64));
        q18_count_big_sorted_runs_kernel<<<blocks_n, BLOCK_SIZE, 0, stream>>>(
            orderkeys.data_ptr<int64_t>(),
            quantity.data_ptr<double>(),
            threshold,
            (unsigned long long*)count_tensor.data_ptr<int64_t>(),
            n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        int64_t out_n = count_tensor.item<int64_t>();
        auto out_keys = torch::empty({out_n}, key_opts);
        auto out_sums = torch::empty({out_n}, qty_opts);

        if (out_n > 0) {
            auto pos_tensor = torch::zeros({1}, orderkeys.options().dtype(torch::kInt64));
            q18_fill_big_sorted_runs_kernel<<<blocks_n, BLOCK_SIZE, 0, stream>>>(
                orderkeys.data_ptr<int64_t>(),
                quantity.data_ptr<double>(),
                threshold,
                out_keys.data_ptr<int64_t>(),
                out_sums.data_ptr<double>(),
                (unsigned long long*)pos_tensor.data_ptr<int64_t>(),
                n
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        return {out_keys, out_sums};
    }

    int64_t max_key = max_tensor.item<int64_t>();
    TORCH_CHECK(max_key >= 0, "negative order keys are not supported by dense fallback");

    int64_t dense_n = max_key + 1;
    auto sums = torch::zeros({dense_n}, qty_opts);

    q18_atomic_aggregate_kernel<<<blocks_n, BLOCK_SIZE, 0, stream>>>(
        orderkeys.data_ptr<int64_t>(),
        quantity.data_ptr<double>(),
        sums.data_ptr<double>(),
        n,
        dense_n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_dense = pick_blocks(dense_n);
    auto count_tensor = torch::zeros({1}, orderkeys.options().dtype(torch::kInt64));
    q18_count_big_dense_kernel<<<blocks_dense, BLOCK_SIZE, 0, stream>>>(
        sums.data_ptr<double>(),
        threshold,
        (unsigned long long*)count_tensor.data_ptr<int64_t>(),
        dense_n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t out_n = count_tensor.item<int64_t>();
    auto out_keys = torch::empty({out_n}, key_opts);
    auto out_sums = torch::empty({out_n}, qty_opts);

    if (out_n > 0) {
        auto pos_tensor = torch::zeros({1}, orderkeys.options().dtype(torch::kInt64));
        q18_fill_big_dense_kernel<<<blocks_dense, BLOCK_SIZE, 0, stream>>>(
            sums.data_ptr<double>(),
            threshold,
            out_keys.data_ptr<int64_t>(),
            out_sums.data_ptr<double>(),
            (unsigned long long*)pos_tensor.data_ptr<int64_t>(),
            dense_n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {out_keys, out_sums};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q18_aggregate_launcher", &q18_aggregate_launcher, "TPC-H Q18 lineitem quantity aggregation (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension from the in-module .cu string at import time."""
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
    GPU/CUDA hot path for TPC-H Q18's HAVING subquery.

    The CUDA extension aggregates l_quantity by l_orderkey and returns only
    orders whose sum exceeds qty_threshold. For the normal TPC-H physical order,
    lineitem is nondecreasing by l_orderkey, so the kernel uses run-length
    aggregation with coalesced reads and no per-row global atomics. If input is
    not sorted, it falls back to a dense direct-address CUDA aggregation, which
    preserves correctness for arbitrary order.
    """
    big_keys, big_sums = _EXT.q18_aggregate_launcher(
        l_orderkey.contiguous(),
        l_quantity.contiguous(),
        float(qty_threshold),
    )
    return big_keys, big_sums, big_keys


def run_query(
    tables: dict[str, cudf.DataFrame],
    qty_threshold: float = 300.0,
    limit: int = 100,
) -> cudf.DataFrame:
    """
    SQL:
      SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
      FROM customer, orders, lineitem
      WHERE o_orderkey IN (
          SELECT l_orderkey
          FROM lineitem
          GROUP BY l_orderkey
          HAVING sum(l_quantity) > 300)
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
      GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
      ORDER BY o_totalprice DESC, o_orderdate
      LIMIT 100;

    Description:
      TPC-H Q18 — finds the top customers with large-quantity orders. The bulk
      lineitem GROUP BY/HAVING step runs in hand-written CUDA. It is faster than
      the reference torch.unique/scatter path because it avoids a global sort for
      the common sorted TPC-H lineitem layout, using fused run aggregation over
      coalesced GPU reads. cuDF is then used for the relational joins, string
      columns, final ordering, and result DataFrame construction.

    Params:
      tables: cuDF input tables keyed by table name.
      qty_threshold: HAVING threshold for sum(l_quantity).
      limit: number of ordered rows to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    big_keys_t, big_sums_t, _ = _query_core(l_orderkey_t, l_quantity_t, float(qty_threshold))

    df_big = cudf.DataFrame(
        {
            "o_orderkey": cudf.Series(big_keys_t),
            "sum_qty": cudf.Series(big_sums_t),
        }
    )

    orders_cols = df_orders[["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"]]
    orders_filtered = orders_cols.merge(df_big, on="o_orderkey", how="inner")

    joined = orders_filtered.merge(
        df_customer[["c_custkey", "c_name"]],
        left_on="o_custkey",
        right_on="c_custkey",
        how="inner",
    )

    joined = joined.sort_values(
        by=["o_totalprice", "o_orderdate"],
        ascending=[False, True],
    ).head(limit)

    return cudf.DataFrame(
        {
            "c_name": joined["c_name"].reset_index(drop=True),
            "c_custkey": joined["c_custkey"].reset_index(drop=True),
            "o_orderkey": joined["o_orderkey"].reset_index(drop=True),
            "o_orderdate": joined["o_orderdate"].reset_index(drop=True),
            "o_totalprice": joined["o_totalprice"].reset_index(drop=True),
            "sum(l_quantity)": joined["sum_qty"].reset_index(drop=True),
        }
    )
