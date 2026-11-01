import os
import tempfile
from typing import Any, Tuple

import cudf
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <vector>

#define BLOCK_SIZE 1024

__device__ __forceinline__ int find_code_idx(
    int32_t code,
    const int32_t* __restrict__ codes,
    int num_codes
) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= num_codes) break;
        if (codes[i] == code) return i;
    }
    for (int i = 16; i < num_codes; ++i) {
        if (codes[i] == code) return i;
    }
    return -1;
}

__global__ void build_order_flags_kernel(
    const int64_t* __restrict__ o_custkey,
    uint8_t* __restrict__ flags,
    int64_t n_orders,
    int64_t flags_len
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t key = o_custkey[idx];
        if (key >= 0 && key < flags_len) {
            flags[key] = static_cast<uint8_t>(1);
        }
    }
}

__global__ void avg_acctbal_kernel(
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const int32_t* __restrict__ codes,
    int num_codes,
    double* __restrict__ avg_stats,
    int64_t n_customers
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ double block_count[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double my_sum = 0.0;
    double my_count = 0.0;

    if (idx < n_customers) {
        double bal = c_acctbal[idx];
        if (bal > 0.0) {
            int32_t code = cntrycode[idx];
            if (find_code_idx(code, codes, num_codes) >= 0) {
                my_sum = bal;
                my_count = 1.0;
            }
        }
    }

    block_sum[threadIdx.x] = my_sum;
    block_count[threadIdx.x] = my_count;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
            block_count[threadIdx.x] += block_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&avg_stats[0], block_sum[0]);
        atomicAdd(&avg_stats[1], block_count[0]);
    }
}

__global__ void final_group_kernel(
    const int64_t* __restrict__ c_custkey,
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const uint8_t* __restrict__ order_flags,
    int64_t flags_len,
    const int32_t* __restrict__ codes,
    int num_codes,
    const double* __restrict__ avg_stats,
    int64_t* __restrict__ out_counts_i64,
    double* __restrict__ out_sums,
    int64_t n_customers
) {
    extern __shared__ unsigned char shmem[];
    unsigned long long* sh_counts = reinterpret_cast<unsigned long long*>(shmem);
    double* sh_sums = reinterpret_cast<double*>(shmem + num_codes * sizeof(unsigned long long));

    for (int i = threadIdx.x; i < num_codes; i += blockDim.x) {
        sh_counts[i] = 0ULL;
        sh_sums[i] = 0.0;
    }
    __syncthreads();

    double avg_val = avg_stats[0] / avg_stats[1];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_customers) {
        double bal = c_acctbal[idx];
        if (bal > avg_val) {
            int32_t code = cntrycode[idx];
            int code_idx = find_code_idx(code, codes, num_codes);
            if (code_idx >= 0) {
                int64_t custkey = c_custkey[idx];
                bool no_order = true;
                if (custkey >= 0 && custkey < flags_len) {
                    no_order = (order_flags[custkey] == static_cast<uint8_t>(0));
                }
                if (no_order) {
                    atomicAdd(&sh_counts[code_idx], 1ULL);
                    atomicAdd(&sh_sums[code_idx], bal);
                }
            }
        }
    }
    __syncthreads();

    unsigned long long* out_counts = reinterpret_cast<unsigned long long*>(out_counts_i64);
    for (int i = threadIdx.x; i < num_codes; i += blockDim.x) {
        unsigned long long cnt = sh_counts[i];
        double sum = sh_sums[i];
        if (cnt != 0ULL) {
            atomicAdd(&out_counts[i], cnt);
            atomicAdd(&out_sums[i], sum);
        }
    }
}

std::vector<torch::Tensor> q22_execute(
    torch::Tensor c_custkey,
    torch::Tensor c_acctbal,
    torch::Tensor cntrycode,
    torch::Tensor o_custkey,
    torch::Tensor codes,
    int64_t flags_len
) {
    TORCH_CHECK(c_custkey.is_cuda(), "c_custkey must be CUDA");
    TORCH_CHECK(c_acctbal.is_cuda(), "c_acctbal must be CUDA");
    TORCH_CHECK(cntrycode.is_cuda(), "cntrycode must be CUDA");
    TORCH_CHECK(o_custkey.is_cuda(), "o_custkey must be CUDA");
    TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");

    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64, "c_custkey must be int64");
    TORCH_CHECK(c_acctbal.scalar_type() == torch::kFloat64, "c_acctbal must be float64");
    TORCH_CHECK(cntrycode.scalar_type() == torch::kInt32, "cntrycode must be int32");
    TORCH_CHECK(o_custkey.scalar_type() == torch::kInt64, "o_custkey must be int64");
    TORCH_CHECK(codes.scalar_type() == torch::kInt32, "codes must be int32");

    c_custkey = c_custkey.contiguous();
    c_acctbal = c_acctbal.contiguous();
    cntrycode = cntrycode.contiguous();
    o_custkey = o_custkey.contiguous();
    codes = codes.contiguous();

    int64_t n_customers = c_acctbal.numel();
    int64_t n_orders = o_custkey.numel();
    int num_codes = static_cast<int>(codes.numel());

    TORCH_CHECK(c_custkey.numel() == n_customers, "customer key/account length mismatch");
    TORCH_CHECK(cntrycode.numel() == n_customers, "country/account length mismatch");
    TORCH_CHECK(flags_len > 0, "flags_len must be positive");
    TORCH_CHECK(num_codes <= 1024, "too many country codes for this Q22 CUDA kernel");

    auto count_opts = c_custkey.options().dtype(torch::kInt64);
    auto double_opts = c_acctbal.options().dtype(torch::kFloat64);
    auto byte_opts = c_custkey.options().dtype(torch::kUInt8);

    torch::Tensor out_counts = torch::zeros({num_codes}, count_opts);
    torch::Tensor out_sums = torch::zeros({num_codes}, double_opts);

    if (num_codes == 0 || n_customers == 0) {
        return {out_counts, out_sums};
    }

    torch::Tensor order_flags = torch::zeros({flags_len}, byte_opts);
    torch::Tensor avg_stats = torch::zeros({2}, double_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n_orders > 0) {
        int order_blocks = static_cast<int>((n_orders + BLOCK_SIZE - 1) / BLOCK_SIZE);
        build_order_flags_kernel<<<order_blocks, BLOCK_SIZE, 0, stream>>>(
            o_custkey.data_ptr<int64_t>(),
            order_flags.data_ptr<uint8_t>(),
            n_orders,
            flags_len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int customer_blocks = static_cast<int>((n_customers + BLOCK_SIZE - 1) / BLOCK_SIZE);

    avg_acctbal_kernel<<<customer_blocks, BLOCK_SIZE, 0, stream>>>(
        c_acctbal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        codes.data_ptr<int32_t>(),
        num_codes,
        avg_stats.data_ptr<double>(),
        n_customers
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int shared_bytes = num_codes * (static_cast<int>(sizeof(unsigned long long)) + static_cast<int>(sizeof(double)));
    final_group_kernel<<<customer_blocks, BLOCK_SIZE, shared_bytes, stream>>>(
        c_custkey.data_ptr<int64_t>(),
        c_acctbal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        order_flags.data_ptr<uint8_t>(),
        flags_len,
        codes.data_ptr<int32_t>(),
        num_codes,
        avg_stats.data_ptr<double>(),
        out_counts.data_ptr<int64_t>(),
        out_sums.data_ptr<double>(),
        n_customers
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out_counts, out_sums};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q22_execute", &q22_execute, "TPC-H Q22 fused order-flag, average, and grouped aggregate CUDA pipeline");
}
'''


def _load_cuda() -> Any:
    """Compile the embedded CUDA/C++ extension from a temporary on-disk .cu file."""
    tmpdir = tempfile.mkdtemp(prefix="q22_cuda_")
    cu_path = os.path.join(tmpdir, "_q22_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q22_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT: Any = _load_cuda()


def _empty_result() -> cudf.DataFrame:
    """Create an empty Q22 result with the reference column names and dtypes on the cuDF side."""
    return cudf.DataFrame(
        {
            "cntrycode": cudf.Series([], dtype="str"),
            "numcust": cudf.Series([], dtype="int64"),
            "totacctbal": cudf.Series([], dtype="float64"),
        }
    )


def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    o_custkey: torch.Tensor,
    codes: torch.Tensor,
    flags_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the Q22 numeric hot path on the GPU using hand-written CUDA.

    The CUDA extension builds a dense order-existence bitmap, computes the positive-balance
    country-filtered average, and performs the final NOT EXISTS/filter/grouped aggregation.
    This replaces the reference sort/searchsorted/masking/scatter pipeline with three
    coalesced CUDA passes and small fixed-group reductions, avoiding large temporary masks
    and an expensive sort of orders.
    """
    counts, sums = _EXT.q22_execute(c_custkey, c_acctbal, cntrycode, o_custkey, codes, int(flags_len))
    return counts, sums


def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: tuple[str, ...] = ("13", "31", "23", "29", "30", "18", "17"),
) -> cudf.DataFrame:
    """
    SQL: SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal
         FROM (SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
               FROM customer
               WHERE substring(c_phone FROM 1 FOR 2) IN (...)
                 AND c_acctbal > (SELECT avg(c_acctbal) FROM customer
                                  WHERE c_acctbal > 0.00
                                    AND substring(c_phone FROM 1 FOR 2) IN (...))
                 AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
              ) AS custsale
         GROUP BY cntrycode ORDER BY cntrycode;

    Description: TPC-H Q22. cuDF handles string extraction of the first two phone digits.
                 The bulk numeric hot path runs in hand-written CUDA: a dense customer-key
                 order bitmap replaces sorting/searching orders, then fused CUDA scans compute
                 the average and final fixed-country grouped aggregates with fewer temporaries
                 and less Python/cuDF/PyTorch orchestration than the reference plan.

    Params:
        tables: Dictionary containing cuDF DataFrames "customer" and "orders".
        cntrycodes: Tuple of two-digit country-code strings used by the SQL IN predicate.
    """
    codes_sorted = sorted(set(int(code) for code in cntrycodes))
    if len(codes_sorted) == 0:
        return _empty_result()

    df_customer = tables["customer"]
    df_orders = tables["orders"]

    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    o_custkey_t = torch.as_tensor(df_orders["o_custkey"].to_cupy(), dtype=torch.int64)
    codes_t = torch.tensor(codes_sorted, dtype=torch.int32, device=c_acctbal_t.device)

    max_custkey = int(df_customer["c_custkey"].max()) if len(df_customer) else 0
    flags_len = max(1, max_custkey + 1)

    numcust_t, totacctbal_t = _query_core(
        c_custkey_t,
        c_acctbal_t,
        cntrycode_t,
        o_custkey_t,
        codes_t,
        flags_len,
    )

    torch.cuda.synchronize(device=c_acctbal_t.device)

    result = cudf.DataFrame(
        {
            "cntrycode": cudf.Series(codes_t).astype("int32").astype("str").str.zfill(2),
            "numcust": cudf.Series(numcust_t),
            "totacctbal": cudf.Series(totacctbal_t),
        }
    )
    result = result[result["numcust"] > 0].reset_index(drop=True)
    return result
