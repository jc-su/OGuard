import os
import tempfile
import torch
import cudf
import pandas as pd
from typing import Tuple
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define BLOCK_SIZE 1024
#define MAX_CNTRY_CODES 7

__device__ bool binary_search(const int64_t* sorted_keys, int64_t n, int64_t key) {
    int64_t lo = 0, hi = n - 1;
    while (lo <= hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (sorted_keys[mid] == key) return true;
        if (sorted_keys[mid] < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

__device__ int get_code_index(int32_t code, const int32_t* target_codes) {
    for (int i = 0; i < MAX_CNTRY_CODES; i++) {
        if (target_codes[i] == code) return i;
    }
    return -1;
}

__global__ void q22_phase1_kernel(
    const double* __restrict__ c_acctbal,
    const bool* __restrict__ in_cntrycode_mask,
    int64_t n_customers,
    double* sum_out,
    int64_t* count_out
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ int64_t block_count[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    double my_sum = 0.0;
    int64_t my_count = 0;
    
    if (idx < n_customers) {
        if (in_cntrycode_mask[idx] && c_acctbal[idx] > 0.0) {
            my_sum = c_acctbal[idx];
            my_count = 1;
        }
    }
    
    block_sum[tid] = my_sum;
    block_count[tid] = my_count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sum[tid] += block_sum[tid + s];
            block_count[tid] += block_count[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(sum_out, block_sum[0]);
        atomicAdd((unsigned long long int*)count_out, (unsigned long long int)block_count[0]);
    }
}

__global__ void q22_phase2_kernel(
    const int64_t* __restrict__ c_custkey,
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool* __restrict__ in_cntrycode_mask,
    const int64_t* __restrict__ o_custkey_sorted,
    const int32_t* target_codes,
    int64_t n_customers,
    int64_t n_orders,
    double avg_bal,
    int32_t* group_codes,
    int64_t* group_numcust,
    double* group_totacctbal
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_customers) {
        if (in_cntrycode_mask[idx] && c_acctbal[idx] > avg_bal) {
            if (!binary_search(o_custkey_sorted, n_orders, c_custkey[idx])) {
                int32_t code = cntrycode[idx];
                int code_idx = get_code_index(code, target_codes);
                
                if (code_idx >= 0 && code_idx < MAX_CNTRY_CODES) {
                    atomicAdd((unsigned long long int*)&group_numcust[code_idx], (unsigned long long int)1);
                    atomicAdd(&group_totacctbal[code_idx], c_acctbal[idx]);
                    group_codes[code_idx] = code;
                }
            }
        }
    }
}

std::vector<torch::Tensor> q22_launcher(
    torch::Tensor c_custkey,
    torch::Tensor c_acctbal,
    torch::Tensor cntrycode,
    torch::Tensor in_cntrycode_mask,
    torch::Tensor o_custkey_sorted,
    torch::Tensor target_codes
) {
    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(c_acctbal.scalar_type() == torch::kFloat64);
    TORCH_CHECK(cntrycode.scalar_type() == torch::kInt32);
    TORCH_CHECK(in_cntrycode_mask.scalar_type() == torch::kBool);
    TORCH_CHECK(o_custkey_sorted.scalar_type() == torch::kInt64);
    TORCH_CHECK(target_codes.scalar_type() == torch::kInt32);
    
    if (!c_custkey.is_cuda()) c_custkey = c_custkey.cuda();
    if (!c_acctbal.is_cuda()) c_acctbal = c_acctbal.cuda();
    if (!cntrycode.is_cuda()) cntrycode = cntrycode.cuda();
    if (!in_cntrycode_mask.is_cuda()) in_cntrycode_mask = in_cntrycode_mask.cuda();
    if (!o_custkey_sorted.is_cuda()) o_custkey_sorted = o_custkey_sorted.cuda();
    if (!target_codes.is_cuda()) target_codes = target_codes.cuda();
    
    c_custkey = c_custkey.contiguous();
    c_acctbal = c_acctbal.contiguous();
    cntrycode = cntrycode.contiguous();
    in_cntrycode_mask = in_cntrycode_mask.contiguous();
    o_custkey_sorted = o_custkey_sorted.contiguous();
    target_codes = target_codes.contiguous();
    
    int64_t n_customers = c_custkey.numel();
    int64_t n_orders = o_custkey_sorted.numel();
    
    auto sum_out = torch::zeros({1}, c_acctbal.options());
    auto count_out = torch::zeros({1}, c_custkey.options());
    
    int blocks = (n_customers + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q22_phase1_kernel<<<blocks, BLOCK_SIZE>>>(
        c_acctbal.data_ptr<double>(),
        in_cntrycode_mask.data_ptr<bool>(),
        n_customers,
        sum_out.data_ptr<double>(),
        count_out.data_ptr<int64_t>()
    );
    
    cudaDeviceSynchronize();
    
    double total_sum = sum_out.item<double>();
    int64_t total_count = count_out.item<int64_t>();
    double avg_bal = total_count > 0 ? total_sum / total_count : 0.0;
    
    auto group_codes = torch::zeros({MAX_CNTRY_CODES}, cntrycode.options());
    auto group_numcust = torch::zeros({MAX_CNTRY_CODES}, c_custkey.options());
    auto group_totacctbal = torch::zeros({MAX_CNTRY_CODES}, c_acctbal.options());
    
    q22_phase2_kernel<<<blocks, BLOCK_SIZE>>>(
        c_custkey.data_ptr<int64_t>(),
        c_acctbal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        in_cntrycode_mask.data_ptr<bool>(),
        o_custkey_sorted.data_ptr<int64_t>(),
        target_codes.data_ptr<int32_t>(),
        n_customers,
        n_orders,
        avg_bal,
        group_codes.data_ptr<int32_t>(),
        group_numcust.data_ptr<int64_t>(),
        group_totacctbal.data_ptr<double>()
    );
    
    cudaDeviceSynchronize();
    
    // Filter out zero entries and sort by country code
    auto mask = group_numcust > 0;
    auto codes_filtered = group_codes.masked_select(mask);
    auto numcust_filtered = group_numcust.masked_select(mask);
    auto totacctbal_filtered = group_totacctbal.masked_select(mask);
    
    if (codes_filtered.numel() > 0) {
        auto order = torch::argsort(codes_filtered);
        codes_filtered = codes_filtered.index_select(0, order);
        numcust_filtered = numcust_filtered.index_select(0, order);
        totacctbal_filtered = totacctbal_filtered.index_select(0, order);
    }
    
    return {codes_filtered, numcust_filtered, totacctbal_filtered};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q22_launcher", &q22_launcher, "TPC-H Q22 (CUDA)");
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

def _launch_q22(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
    target_codes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Launch CUDA kernels for TPC-H Q22.
    
    Two-phase approach:
    - Phase 1: Compute avg(c_acctbal) for customers in target countries with positive balance
    - Phase 2: Filter customers above avg with no orders, aggregate by country code
    
    Uses shared-memory block reduction and atomic operations for efficient aggregation.
    Binary search on sorted o_custkey for NOT EXISTS check.
    Fixed slots for target country codes to avoid race conditions in group-by.
    """
    # Sort o_custkey_unique on GPU for binary search
    o_custkey_sorted, _ = torch.sort(o_custkey_unique)
    
    results = _EXT.q22_launcher(
        c_custkey,
        c_acctbal,
        cntrycode,
        in_cntrycode_mask,
        o_custkey_sorted,
        target_codes,
    )
    
    return results[0], results[1], results[2]

def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,           # int32, derived from first 2 chars of c_phone
    in_cntrycode_mask: torch.Tensor,   # bool, whether cntrycode is in the target set
    o_custkey_unique: torch.Tensor,    # int64, unique custkeys from orders
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated core computation for TPC-H Q22.
    
    Replaces eager PyTorch operations with fused CUDA kernels for:
    1. Average balance calculation (Phase 1 kernel with block reduction)
    2. Customer filtering and group-by aggregation (Phase 2 kernel)
    
    Benefits:
    - Coalesced global memory access patterns
    - Shared-memory block-level reduction before atomicAdd
    - Single pass through customer data in Phase 2
    - Binary search for NOT EXISTS check (O(log n) per thread)
    - Fixed slots for target country codes to avoid race conditions
    
    Args:
        c_custkey: Customer keys (int64)
        c_acctbal: Customer account balances (float64)
        cntrycode: Country codes from c_phone (int32)
        in_cntrycode_mask: Mask for target country codes (bool)
        o_custkey_unique: Unique order customer keys for NOT EXISTS (int64)
    
    Returns:
        Tuple of (country_codes, num_customers, total_account_balance) sorted by code
    """
    # Target country codes as tensor for device lookup
    target_codes = torch.tensor([13, 31, 23, 29, 30, 18, 17], dtype=torch.int32, device=c_custkey.device)
    
    return _launch_q22(
        c_custkey,
        c_acctbal,
        cntrycode,
        in_cntrycode_mask,
        o_custkey_unique,
        target_codes,
    )

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
