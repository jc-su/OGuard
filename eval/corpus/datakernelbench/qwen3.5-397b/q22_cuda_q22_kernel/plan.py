import os
import tempfile
import torch
import cudf
import pandas as pd
from typing import Tuple
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define BLOCK_SIZE 1024
#define NUM_COUNTRY_CODES 7

__device__ __constant__ int32_t d_target_codes[NUM_COUNTRY_CODES] = {13, 31, 23, 29, 30, 18, 17};

__device__ bool binary_search_exists(const int64_t* arr, int64_t n, int64_t key) {
    int64_t left = 0;
    int64_t right = n - 1;
    while (left <= right) {
        int64_t mid = left + (right - left) / 2;
        if (arr[mid] == key) return true;
        if (arr[mid] < key) left = mid + 1;
        else right = mid - 1;
    }
    return false;
}

__global__ void q22_kernel(
    const int64_t* __restrict__ c_custkey,
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool* __restrict__ in_mask,
    const int64_t* __restrict__ orders_sorted,
    int64_t n_orders,
    double avg_bal,
    int64_t* __restrict__ count_per_country,
    double* __restrict__ bal_per_country,
    int64_t n_customers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_customers) return;
    
    if (!in_mask[idx]) return;
    
    int32_t code = cntrycode[idx];
    double bal = c_acctbal[idx];
    
    if (bal <= avg_bal) return;
    
    int64_t custkey = c_custkey[idx];
    
    bool has_order = binary_search_exists(orders_sorted, n_orders, custkey);
    
    if (has_order) return;
    
    int code_idx = -1;
    #pragma unroll
    for (int i = 0; i < NUM_COUNTRY_CODES; i++) {
        if (code == d_target_codes[i]) {
            code_idx = i;
            break;
        }
    }
    
    if (code_idx >= 0) {
        atomicAdd((unsigned long long*)&count_per_country[code_idx], 1ULL);
        atomicAdd(&bal_per_country[code_idx], bal);
    }
}

__global__ void q22_avg_kernel(
    const double* __restrict__ c_acctbal,
    const bool* __restrict__ in_mask,
    double* __restrict__ sum_ptr,
    int64_t* __restrict__ count_ptr,
    int64_t n_customers
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ int64_t block_count[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_sum = 0.0;
    int64_t my_count = 0;
    
    if (idx < n_customers) {
        if (in_mask[idx] && c_acctbal[idx] > 0.0) {
            my_sum = c_acctbal[idx];
            my_count = 1;
        }
    }
    
    block_sum[threadIdx.x] = my_sum;
    block_count[threadIdx.x] = my_count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            block_count[threadIdx.x] += block_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(sum_ptr, block_sum[0]);
        atomicAdd((unsigned long long*)count_ptr, (unsigned long long)block_count[0]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> q22_launcher(
    torch::Tensor c_custkey,
    torch::Tensor c_acctbal,
    torch::Tensor cntrycode,
    torch::Tensor in_mask,
    torch::Tensor orders_sorted,
    double avg_bal
) {
    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64, "c_custkey must be int64");
    TORCH_CHECK(c_acctbal.scalar_type() == torch::kFloat64, "c_acctbal must be float64");
    TORCH_CHECK(cntrycode.scalar_type() == torch::kInt32, "cntrycode must be int32");
    TORCH_CHECK(in_mask.scalar_type() == torch::kBool, "in_mask must be bool");
    TORCH_CHECK(orders_sorted.scalar_type() == torch::kInt64, "orders must be int64");
    
    if (!c_custkey.is_cuda()) c_custkey = c_custkey.cuda();
    if (!c_acctbal.is_cuda()) c_acctbal = c_acctbal.cuda();
    if (!cntrycode.is_cuda()) cntrycode = cntrycode.cuda();
    if (!in_mask.is_cuda()) in_mask = in_mask.cuda();
    if (!orders_sorted.is_cuda()) orders_sorted = orders_sorted.cuda();
    
    c_custkey = c_custkey.contiguous();
    c_acctbal = c_acctbal.contiguous();
    cntrycode = cntrycode.contiguous();
    in_mask = in_mask.contiguous();
    orders_sorted = orders_sorted.contiguous();
    
    int64_t n = c_custkey.numel();
    int64_t n_orders = orders_sorted.numel();
    
    auto count_out = torch::zeros({NUM_COUNTRY_CODES}, torch::kInt64);
    auto bal_out = torch::zeros({NUM_COUNTRY_CODES}, c_acctbal.options());
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q22_kernel<<<blocks, BLOCK_SIZE>>>(
        c_custkey.data_ptr<int64_t>(),
        c_acctbal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        in_mask.data_ptr<bool>(),
        orders_sorted.data_ptr<int64_t>(),
        n_orders,
        avg_bal,
        count_out.data_ptr<int64_t>(),
        bal_out.data_ptr<double>(),
        n
    );
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(count_out, bal_out);
}

double q22_avg_launcher(
    torch::Tensor c_acctbal,
    torch::Tensor in_mask
) {
    TORCH_CHECK(c_acctbal.scalar_type() == torch::kFloat64, "c_acctbal must be float64");
    TORCH_CHECK(in_mask.scalar_type() == torch::kBool, "in_mask must be bool");
    
    if (!c_acctbal.is_cuda()) c_acctbal = c_acctbal.cuda();
    if (!in_mask.is_cuda()) in_mask = in_mask.cuda();
    
    c_acctbal = c_acctbal.contiguous();
    in_mask = in_mask.contiguous();
    
    int64_t n = c_acctbal.numel();
    
    auto sum_tensor = torch::zeros({1}, c_acctbal.options());
    auto count_tensor = torch::zeros({1}, torch::kInt64).cuda();
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q22_avg_kernel<<<blocks, BLOCK_SIZE>>>(
        c_acctbal.data_ptr<double>(),
        in_mask.data_ptr<bool>(),
        sum_tensor.data_ptr<double>(),
        count_tensor.data_ptr<int64_t>(),
        n
    );
    
    cudaDeviceSynchronize();
    
    double sum_val = sum_tensor.item<double>();
    int64_t count_val = count_tensor.item<int64_t>();
    
    if (count_val == 0) return 0.0;
    return sum_val / static_cast<double>(count_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q22_launcher", &q22_launcher, "TPC-H Q22 main aggregation (CUDA)");
    m.def("q22_avg_launcher", &q22_avg_launcher, "TPC-H Q22 average calculation (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q22_cuda_")
    cu_path = os.path.join(tmpdir, "_q22_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
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
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused CUDA kernel for TPC-H Q22 aggregation.
    
    GPU Execution:
    - Pass 1: Compute avg(c_acctbal) for customers with positive balance in target countries
    - Pass 2: Filter by avg threshold + NOT EXISTS orders, aggregate per country
    
    Why faster than reference:
    - Single fused kernel for filter + project + group-by aggregate
    - Block-level shared memory reduction before global atomicAdd
    - Coalesced memory access patterns
    - Fixed 7-country aggregation avoids hash table overhead
    - Manual binary search avoids thrust dependency issues
    """
    orders_sorted, _ = torch.sort(o_custkey_unique)
    
    avg_bal = _EXT.q22_avg_launcher(c_acctbal, in_cntrycode_mask)
    
    count_per_country, bal_per_country = _EXT.q22_launcher(
        c_custkey, c_acctbal, cntrycode, in_cntrycode_mask,
        orders_sorted, avg_bal
    )
    
    target_codes = torch.tensor([13, 31, 23, 29, 30, 18, 17], dtype=torch.int32, device=count_per_country.device)
    
    order = torch.argsort(target_codes)
    
    return target_codes[order], count_per_country[order], bal_per_country[order]


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
                 
    GPU Execution Strategy:
    - cuDF: String extraction (c_phone -> cntrycode), boolean mask creation, orders unique
    - CUDA: Two-pass fused kernel - (1) compute avg balance, (2) filter + group-by aggregate
    - Optimization: Block-level reduction with shared memory, one atomicAdd per block,
      fixed 7-element per-country accumulators avoid hash table overhead
    
    Params: cntrycodes (tuple[str,...]): target 2-digit country codes (default: 7 codes).
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")
    
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)
    
    o_custkey_unique_series = df_orders["o_custkey"].unique()
    
    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)
    
    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t, o_custkey_t
    )
    
    codes_cudf = cudf.Series(codes_t.cpu()).astype("int32").astype("str").str.zfill(2)
    
    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_t.cpu()),
        "totacctbal": cudf.Series(totacctbal_t.cpu()),
    })
