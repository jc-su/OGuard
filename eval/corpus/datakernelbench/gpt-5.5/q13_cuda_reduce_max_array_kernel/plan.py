import os
import re
import tempfile
from typing import Any

import cudf
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 1024

#define CUDA_CHECK_LAST() do {                                      \
    cudaError_t err = cudaGetLastError();                           \
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));        \
} while (0)

__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long int*>(addr),
              static_cast<unsigned long long int>(val));
}

__device__ __forceinline__ int64_t atomic_inc_i64(int64_t* addr) {
    return static_cast<int64_t>(
        atomicAdd(reinterpret_cast<unsigned long long int*>(addr), 1ULL)
    );
}

__global__ void count_orders_kernel(
    const int64_t* __restrict__ o_custkey,
    int64_t n_orders,
    int64_t* __restrict__ counts,
    int64_t counts_len
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = tid; i < n_orders; i += stride) {
        int64_t key = o_custkey[i];
        if (key >= 0 && key < counts_len) {
            atomic_add_i64(counts + key, 1);
        }
    }
}

__global__ void reduce_max_customer_counts_kernel(
    const int64_t* __restrict__ c_custkey,
    const int64_t* __restrict__ counts,
    int64_t n_customers,
    int64_t counts_len,
    int64_t* __restrict__ block_max
) {
    __shared__ int64_t smem[BLOCK_SIZE];

    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t local_max = 0;

    for (int64_t i = tid; i < n_customers; i += stride) {
        int64_t key = c_custkey[i];
        int64_t v = 0;
        if (key >= 0 && key < counts_len) {
            v = counts[key];
        }
        if (v > local_max) {
            local_max = v;
        }
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            int64_t other = smem[threadIdx.x + s];
            if (other > smem[threadIdx.x]) {
                smem[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}

__global__ void reduce_max_array_kernel(
    const int64_t* __restrict__ values,
    int64_t n_values,
    int64_t* __restrict__ block_max
) {
    __shared__ int64_t smem[BLOCK_SIZE];

    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t local_max = 0;

    for (int64_t i = tid; i < n_values; i += stride) {
        int64_t v = values[i];
        if (v > local_max) {
            local_max = v;
        }
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            int64_t other = smem[threadIdx.x + s];
            if (other > smem[threadIdx.x]) {
                smem[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}

__global__ void histogram_customer_counts_kernel(
    const int64_t* __restrict__ c_custkey,
    const int64_t* __restrict__ counts,
    int64_t n_customers,
    int64_t counts_len,
    int64_t* __restrict__ hist,
    int64_t hist_len
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = tid; i < n_customers; i += stride) {
        int64_t key = c_custkey[i];
        int64_t cnt = 0;
        if (key >= 0 && key < counts_len) {
            cnt = counts[key];
        }
        if (cnt >= 0 && cnt < hist_len) {
            atomic_add_i64(hist + cnt, 1);
        }
    }
}

__global__ void compact_histogram_kernel(
    const int64_t* __restrict__ hist,
    int64_t hist_len,
    int64_t* __restrict__ unique_counts,
    int64_t* __restrict__ custdist,
    int64_t* __restrict__ n_unique
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = tid; i < hist_len; i += stride) {
        int64_t v = hist[i];
        if (v > 0) {
            int64_t pos = atomic_inc_i64(n_unique);
            unique_counts[pos] = i;
            custdist[pos] = v;
        }
    }
}

static inline int launch_blocks_for(int64_t n) {
    if (n <= 0) {
        return 1;
    }
    int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = std::min<int64_t>(blocks, 131072);
    return static_cast<int>(blocks);
}

std::vector<torch::Tensor> q13_launcher(
    torch::Tensor c_custkey,
    torch::Tensor o_custkey,
    int64_t num_customers_max_key
) {
    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64, "c_custkey must be int64");
    TORCH_CHECK(o_custkey.scalar_type() == torch::kInt64, "o_custkey must be int64");
    TORCH_CHECK(c_custkey.is_cuda(), "c_custkey must be a CUDA tensor");
    TORCH_CHECK(o_custkey.is_cuda(), "o_custkey must be a CUDA tensor");
    TORCH_CHECK(c_custkey.device() == o_custkey.device(), "input tensors must be on the same CUDA device");
    TORCH_CHECK(num_customers_max_key >= 0, "num_customers_max_key must be non-negative");

    c_custkey = c_custkey.contiguous();
    o_custkey = o_custkey.contiguous();

    const int64_t n_customers = c_custkey.numel();
    const int64_t n_orders = o_custkey.numel();
    const int64_t counts_len = num_customers_max_key + 1;

    auto opts_i64 = o_custkey.options().dtype(torch::kInt64);
    torch::Tensor counts = torch::zeros({counts_len}, opts_i64);

    if (n_orders > 0) {
        int blocks_orders = launch_blocks_for(n_orders);
        count_orders_kernel<<<blocks_orders, BLOCK_SIZE>>>(
            o_custkey.data_ptr<int64_t>(),
            n_orders,
            counts.data_ptr<int64_t>(),
            counts_len
        );
        CUDA_CHECK_LAST();
    }

    int64_t max_count = 0;
    if (n_customers > 0) {
        int first_blocks = launch_blocks_for(n_customers);
        torch::Tensor cur = torch::empty({first_blocks}, opts_i64);

        reduce_max_customer_counts_kernel<<<first_blocks, BLOCK_SIZE>>>(
            c_custkey.data_ptr<int64_t>(),
            counts.data_ptr<int64_t>(),
            n_customers,
            counts_len,
            cur.data_ptr<int64_t>()
        );
        CUDA_CHECK_LAST();

        int64_t cur_n = first_blocks;
        while (cur_n > 1) {
            int next_blocks = launch_blocks_for(cur_n);
            torch::Tensor next = torch::empty({next_blocks}, opts_i64);
            reduce_max_array_kernel<<<next_blocks, BLOCK_SIZE>>>(
                cur.data_ptr<int64_t>(),
                cur_n,
                next.data_ptr<int64_t>()
            );
            CUDA_CHECK_LAST();
            cur = next;
            cur_n = next_blocks;
        }

        max_count = cur.item<int64_t>();
    }

    const int64_t hist_len = max_count + 1;
    torch::Tensor hist = torch::zeros({hist_len}, opts_i64);

    if (n_customers > 0) {
        int blocks_customers = launch_blocks_for(n_customers);
        histogram_customer_counts_kernel<<<blocks_customers, BLOCK_SIZE>>>(
            c_custkey.data_ptr<int64_t>(),
            counts.data_ptr<int64_t>(),
            n_customers,
            counts_len,
            hist.data_ptr<int64_t>(),
            hist_len
        );
        CUDA_CHECK_LAST();
    }

    torch::Tensor unique_counts = torch::empty({hist_len}, opts_i64);
    torch::Tensor custdist = torch::empty({hist_len}, opts_i64);
    torch::Tensor n_unique_t = torch::zeros({1}, opts_i64);

    int blocks_hist = launch_blocks_for(hist_len);
    compact_histogram_kernel<<<blocks_hist, BLOCK_SIZE>>>(
        hist.data_ptr<int64_t>(),
        hist_len,
        unique_counts.data_ptr<int64_t>(),
        custdist.data_ptr<int64_t>(),
        n_unique_t.data_ptr<int64_t>()
    );
    CUDA_CHECK_LAST();

    int64_t n_unique = n_unique_t.item<int64_t>();

    unique_counts = unique_counts.narrow(0, 0, n_unique).contiguous();
    custdist = custdist.narrow(0, 0, n_unique).contiguous();

    return {unique_counts, custdist};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q13_launcher", &q13_launcher, "TPC-H Q13 customer count distribution (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension once; all hot aggregation kernels live in the embedded .cu source."""
    tmpdir = tempfile.mkdtemp(prefix="q13_cuda_")
    cu_path = os.path.join(tmpdir, "_q13_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q13_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _sql_like_to_regex(pattern: str) -> str:
    """Translate a SQL LIKE pattern to a full-match regular expression for cuDF string filtering."""
    pieces: list[str] = []
    for ch in pattern:
        if ch == "%":
            pieces.append(".*")
        elif ch == "_":
            pieces.append(".")
        else:
            pieces.append(re.escape(ch))
    return "^" + "".join(pieces) + "$"


def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the numeric Q13 aggregation on GPU in hand-written CUDA.

    The CUDA extension performs the order-count scatter, max-count reduction,
    customer-count histogram, and histogram compaction on device. This replaces
    the reference PyTorch scatter/gather/unique hot path with fewer large tensor
    temporaries and avoids sorting all customer-level counts; only the tiny final
    distinct-count result is sorted by PyTorch on GPU for SQL order semantics.
    """
    result = _EXT.q13_launcher(c_custkey, o_custkey_filtered, int(num_customers))
    c_count = result[0]
    custdist = result[1]

    if c_count.numel() > 1:
        order1 = torch.argsort(c_count, descending=True, stable=True)
        c_count = c_count[order1]
        custdist = custdist[order1]

        order2 = torch.argsort(custdist, descending=True, stable=True)
        c_count = c_count[order2]
        custdist = custdist[order2]

    return c_count, custdist


def run_query(tables: dict[str, cudf.DataFrame], comment_pattern: str = "%special%requests%") -> cudf.DataFrame:
    """
    SQL:
        SELECT
            c_count,
            count(*) AS custdist
        FROM (
            SELECT
                c_custkey,
                count(o_orderkey)
            FROM
                customer
            LEFT OUTER JOIN orders ON c_custkey = o_custkey
              AND o_comment NOT LIKE '%special%requests%'
            GROUP BY
                c_custkey
        ) AS c_orders (c_custkey, c_count)
        GROUP BY
            c_count
        ORDER BY
            custdist DESC,
            c_count DESC;

    Description:
        TPC-H Q13. cuDF performs the variable-length string predicate on
        o_comment. The filtered order customer keys and customer keys are then
        passed to a CUDA extension that counts orders per customer and builds the
        final customer-count distribution with coalesced scans and compact
        histograms, avoiding PyTorch unique over 1.5M customer rows.

    Params:
        tables: dict[str, cudf.DataFrame]
            Must contain "customer" and "orders".
        comment_pattern: str
            SQL LIKE pattern to exclude from orders. Default is
            '%special%requests%'.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    if len(df_customer) == 0:
        return cudf.DataFrame(
            {
                "c_count": cudf.Series([], dtype="int64"),
                "custdist": cudf.Series([], dtype="int64"),
            }
        )

    if comment_pattern == "%special%requests%":
        regex = "special.*requests"
    else:
        regex = _sql_like_to_regex(comment_pattern)

    mask = ~df_orders["o_comment"].str.contains(regex, regex=True)
    o_custkey_filtered_s = df_orders["o_custkey"][mask]

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    o_custkey_t = torch.as_tensor(o_custkey_filtered_s.to_cupy(), dtype=torch.int64)

    num_customers = int(df_customer["c_custkey"].max())

    c_count_t, custdist_t = _query_core(c_custkey_t, o_custkey_t, num_customers)

    return cudf.DataFrame(
        {
            "c_count": c_count_t,
            "custdist": custdist_t,
        }
    )
