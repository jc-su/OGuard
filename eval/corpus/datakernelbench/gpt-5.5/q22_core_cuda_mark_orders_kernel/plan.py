import os
import tempfile
from typing import Tuple

import torch
import cudf
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 256
#define MAX_CODE 100

__global__ void max_i64_kernel(
    const int64_t* __restrict__ data,
    int64_t* __restrict__ block_max,
    int64_t n
) {
    __shared__ int64_t smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;
    int64_t v = LLONG_MIN;
    if (idx < n) {
        v = data[idx];
    }
    smem[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int64_t other = smem[tid + stride];
            if (other > smem[tid]) {
                smem[tid] = other;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}

__global__ void mark_orders_kernel(
    const int64_t* __restrict__ o_custkey,
    uint8_t* __restrict__ has_order,
    int64_t n_orders,
    int64_t key_space
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t key = o_custkey[idx];
        if (key >= 0 && key < key_space) {
            has_order[key] = 1;
        }
    }
}

__global__ void avg_acctbal_kernel(
    const double* __restrict__ c_acctbal,
    const bool* __restrict__ in_mask,
    double* __restrict__ avg_sum,
    int64_t* __restrict__ avg_count,
    int64_t n
) {
    __shared__ double s_sum[BLOCK_SIZE];
    __shared__ unsigned long long s_count[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;

    double my_sum = 0.0;
    unsigned long long my_count = 0ULL;

    if (idx < n) {
        double bal = c_acctbal[idx];
        if (in_mask[idx] && bal > 0.0) {
            my_sum = bal;
            my_count = 1ULL;
        }
    }

    s_sum[tid] = my_sum;
    s_count[tid] = my_count;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(avg_sum, s_sum[0]);
        atomicAdd(reinterpret_cast<unsigned long long*>(avg_count), s_count[0]);
    }
}

__global__ void final_group_kernel(
    const int64_t* __restrict__ c_custkey,
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool* __restrict__ in_mask,
    const uint8_t* __restrict__ has_order,
    const double* __restrict__ avg_sum,
    const int64_t* __restrict__ avg_count,
    int64_t key_space,
    int64_t* __restrict__ global_counts,
    double* __restrict__ global_sums,
    int64_t n
) {
    __shared__ unsigned long long sh_counts[MAX_CODE];
    __shared__ double sh_sums[MAX_CODE];

    for (int i = threadIdx.x; i < MAX_CODE; i += blockDim.x) {
        sh_counts[i] = 0ULL;
        sh_sums[i] = 0.0;
    }
    __syncthreads();

    unsigned long long cnt = (unsigned long long)avg_count[0];
    double avg = avg_sum[0] / (double)cnt;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && in_mask[idx]) {
        double bal = c_acctbal[idx];
        if (bal > avg) {
            int64_t key = c_custkey[idx];
            bool no_order = true;
            if (key >= 0 && key < key_space) {
                no_order = (has_order[key] == 0);
            }
            if (no_order) {
                int32_t code = cntrycode[idx];
                if (code >= 0 && code < MAX_CODE) {
                    atomicAdd(&sh_counts[code], 1ULL);
                    atomicAdd(&sh_sums[code], bal);
                }
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < MAX_CODE) {
        unsigned long long c = sh_counts[threadIdx.x];
        double s = sh_sums[threadIdx.x];
        if (c != 0ULL) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&global_counts[threadIdx.x]), c);
            atomicAdd(&global_sums[threadIdx.x], s);
        }
    }
}

std::vector<torch::Tensor> q22_launcher(
    torch::Tensor c_custkey,
    torch::Tensor c_acctbal,
    torch::Tensor cntrycode,
    torch::Tensor in_cntrycode_mask,
    torch::Tensor o_custkey_unique
) {
    if (!c_custkey.is_cuda()) c_custkey = c_custkey.to(torch::kCUDA);
    if (!c_acctbal.is_cuda()) c_acctbal = c_acctbal.to(c_custkey.device());
    if (!cntrycode.is_cuda()) cntrycode = cntrycode.to(c_custkey.device());
    if (!in_cntrycode_mask.is_cuda()) in_cntrycode_mask = in_cntrycode_mask.to(c_custkey.device());
    if (!o_custkey_unique.is_cuda()) o_custkey_unique = o_custkey_unique.to(c_custkey.device());

    const c10::cuda::CUDAGuard device_guard(c_custkey.device());

    c_custkey = c_custkey.contiguous();
    c_acctbal = c_acctbal.contiguous();
    cntrycode = cntrycode.contiguous();
    in_cntrycode_mask = in_cntrycode_mask.contiguous();
    o_custkey_unique = o_custkey_unique.contiguous();

    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64, "c_custkey must be int64");
    TORCH_CHECK(c_acctbal.scalar_type() == torch::kFloat64, "c_acctbal must be float64");
    TORCH_CHECK(cntrycode.scalar_type() == torch::kInt32, "cntrycode must be int32");
    TORCH_CHECK(in_cntrycode_mask.scalar_type() == torch::kBool, "in_cntrycode_mask must be bool");
    TORCH_CHECK(o_custkey_unique.scalar_type() == torch::kInt64, "o_custkey_unique must be int64");

    int64_t n_customers = c_custkey.numel();
    int64_t n_orders = o_custkey_unique.numel();

    TORCH_CHECK(c_acctbal.numel() == n_customers, "c_acctbal length mismatch");
    TORCH_CHECK(cntrycode.numel() == n_customers, "cntrycode length mismatch");
    TORCH_CHECK(in_cntrycode_mask.numel() == n_customers, "in_cntrycode_mask length mismatch");

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(c_custkey.device());
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(c_custkey.device());
    auto opts_f64 = torch::TensorOptions().dtype(torch::kFloat64).device(c_custkey.device());

    if (n_customers == 0) {
        return {
            torch::empty({0}, opts_i32),
            torch::empty({0}, opts_i64),
            torch::empty({0}, opts_f64)
        };
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int cust_blocks = (int)((n_customers + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto max_blocks = torch::empty({cust_blocks}, opts_i64);
    max_i64_kernel<<<cust_blocks, BLOCK_SIZE, 0, stream>>>(
        c_custkey.data_ptr<int64_t>(),
        max_blocks.data_ptr<int64_t>(),
        n_customers
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> h_max((size_t)cust_blocks);
    AT_CUDA_CHECK(cudaMemcpyAsync(
        h_max.data(),
        max_blocks.data_ptr<int64_t>(),
        (size_t)cust_blocks * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t max_key = 0;
    for (int i = 0; i < cust_blocks; ++i) {
        if (h_max[(size_t)i] > max_key) {
            max_key = h_max[(size_t)i];
        }
    }
    if (max_key < 0) {
        max_key = 0;
    }
    int64_t key_space = max_key + 1;

    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(c_custkey.device());
    auto has_order = torch::zeros({key_space}, opts_u8);

    if (n_orders > 0) {
        int order_blocks = (int)((n_orders + BLOCK_SIZE - 1) / BLOCK_SIZE);
        mark_orders_kernel<<<order_blocks, BLOCK_SIZE, 0, stream>>>(
            o_custkey_unique.data_ptr<int64_t>(),
            has_order.data_ptr<uint8_t>(),
            n_orders,
            key_space
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto avg_sum = torch::zeros({1}, opts_f64);
    auto avg_count = torch::zeros({1}, opts_i64);

    avg_acctbal_kernel<<<cust_blocks, BLOCK_SIZE, 0, stream>>>(
        c_acctbal.data_ptr<double>(),
        in_cntrycode_mask.data_ptr<bool>(),
        avg_sum.data_ptr<double>(),
        avg_count.data_ptr<int64_t>(),
        n_customers
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto counts = torch::zeros({MAX_CODE}, opts_i64);
    auto sums = torch::zeros({MAX_CODE}, opts_f64);

    final_group_kernel<<<cust_blocks, BLOCK_SIZE, 0, stream>>>(
        c_custkey.data_ptr<int64_t>(),
        c_acctbal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        in_cntrycode_mask.data_ptr<bool>(),
        has_order.data_ptr<uint8_t>(),
        avg_sum.data_ptr<double>(),
        avg_count.data_ptr<int64_t>(),
        key_space,
        counts.data_ptr<int64_t>(),
        sums.data_ptr<double>(),
        n_customers
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> h_counts(MAX_CODE);
    std::vector<double> h_sums(MAX_CODE);

    AT_CUDA_CHECK(cudaMemcpyAsync(
        h_counts.data(),
        counts.data_ptr<int64_t>(),
        MAX_CODE * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    AT_CUDA_CHECK(cudaMemcpyAsync(
        h_sums.data(),
        sums.data_ptr<double>(),
        MAX_CODE * sizeof(double),
        cudaMemcpyDeviceToHost,
        stream
    ));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    int out_n = 0;
    for (int code = 0; code < MAX_CODE; ++code) {
        if (h_counts[code] != 0) {
            ++out_n;
        }
    }

    auto codes_out = torch::empty({out_n}, opts_i32);
    auto nums_out = torch::empty({out_n}, opts_i64);
    auto tots_out = torch::empty({out_n}, opts_f64);

    if (out_n > 0) {
        std::vector<int32_t> h_codes((size_t)out_n);
        std::vector<int64_t> h_nums((size_t)out_n);
        std::vector<double> h_tots((size_t)out_n);

        int j = 0;
        for (int code = 0; code < MAX_CODE; ++code) {
            if (h_counts[code] != 0) {
                h_codes[(size_t)j] = (int32_t)code;
                h_nums[(size_t)j] = h_counts[code];
                h_tots[(size_t)j] = h_sums[code];
                ++j;
            }
        }

        AT_CUDA_CHECK(cudaMemcpyAsync(
            codes_out.data_ptr<int32_t>(),
            h_codes.data(),
            (size_t)out_n * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream
        ));
        AT_CUDA_CHECK(cudaMemcpyAsync(
            nums_out.data_ptr<int64_t>(),
            h_nums.data(),
            (size_t)out_n * sizeof(int64_t),
            cudaMemcpyHostToDevice,
            stream
        ));
        AT_CUDA_CHECK(cudaMemcpyAsync(
            tots_out.data_ptr<double>(),
            h_tots.data(),
            (size_t)out_n * sizeof(double),
            cudaMemcpyHostToDevice,
            stream
        ));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return {codes_out, nums_out, tots_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q22_launcher", &q22_launcher, "TPC-H Q22 fused core using CUDA");
}
'''


def _load_cuda() -> object:
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
    )


_EXT = _load_cuda()


def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,           # int32, derived from first 2 chars of c_phone
    in_cntrycode_mask: torch.Tensor,   # bool, whether cntrycode is in the target set
    o_custkey_unique: torch.Tensor,    # int64, unique custkeys from orders
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs the numeric TPC-H Q22 core on the GPU with handwritten CUDA.

    The CUDA extension replaces eager Torch sort/searchsorted/unique/scatter work with:
    (1) a GPU bitmap build for order existence, (2) a fused GPU scan for the qualifying
    average, and (3) a fused GPU scan for NOT EXISTS filtering plus fixed cntrycode
    aggregation. This avoids sorting millions of order keys and avoids materializing
    filtered customer intermediates, which is faster for this memory-bound scan.
    """
    out = _EXT.q22_launcher(
        c_custkey,
        c_acctbal,
        cntrycode,
        in_cntrycode_mask,
        o_custkey_unique,
    )
    return out[0], out[1], out[2]


def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: tuple[str, ...] = ('13', '31', '23', '29', '30', '18', '17'),
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
    Description: TPC-H Q22 - Global sales opportunity query. Identifies customers in
                 target country codes with above-average balances who have placed no orders,
                 grouped and aggregated by country code.
    Params: cntrycodes (tuple[str,...]): target 2-digit country codes.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Derive cntrycode: first 2 chars of c_phone, convert to int (codes are numeric strings)
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")

    # Build in-set mask on GPU
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)

    # Unique o_custkey from orders (for NOT EXISTS)
    o_custkey_unique_series = df_orders["o_custkey"].unique()

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)

    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t, o_custkey_t
    )

    # Format cntrycode back to 2-char zero-padded string
    codes_cudf = cudf.Series(codes_t).astype("int32").astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_t),
        "totacctbal": cudf.Series(totacctbal_t),
    })
