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

#define BLOCK_SIZE 1024

__global__ void q22_avg_kernel(
    const double* __restrict__ acctbal,
    const bool* __restrict__ mask,
    double* __restrict__ sum_out,
    long long* __restrict__ count_out,
    int64_t n
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ long long block_cnt[BLOCK_SIZE];

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0.0;
    long long my_cnt = 0;

    if (idx < n) {
        bool keep = mask[idx] && (acctbal[idx] > 0.0);
        if (keep) {
            my_sum = acctbal[idx];
            my_cnt = 1;
        }
    }

    block_sum[threadIdx.x] = my_sum;
    block_cnt[threadIdx.x] = my_cnt;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            block_cnt[threadIdx.x] += block_cnt[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum[0]);
        atomicAdd((unsigned long long*)count_out, (unsigned long long)block_cnt[0]);
    }
}

__global__ void q22_filter_kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool* __restrict__ in_mask,
    const int64_t* __restrict__ sorted_okeys,
    int64_t okeys_size,
    double avg_val,
    int32_t* __restrict__ out_code,
    double* __restrict__ out_bal,
    long long* __restrict__ out_n,
    int64_t n
) {
    __shared__ int64_t pred[BLOCK_SIZE];
    __shared__ int32_t s_code[BLOCK_SIZE];
    __shared__ double s_bal[BLOCK_SIZE];
    __shared__ int64_t prefix[BLOCK_SIZE];
    __shared__ long long block_offset;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int32_t my_code = 0;
    double my_bal = 0.0;
    int64_t my_pred = 0;

    if (idx < n) {
        bool in_set = in_mask[idx];
        double bal = acctbal[idx];
        int64_t ck = custkey[idx];

        // Binary search for NOT EXISTS
        int64_t lo = 0;
        int64_t hi = okeys_size;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (sorted_okeys[mid] < ck) lo = mid + 1;
            else hi = mid;
        }
        bool has_order = (lo < okeys_size) && (sorted_okeys[lo] == ck);

        bool keep = in_set && (bal > avg_val) && !has_order;
        if (keep) {
            my_code = cntrycode[idx];
            my_bal = bal;
            my_pred = 1;
        }
    }

    pred[threadIdx.x] = my_pred;
    s_code[threadIdx.x] = my_code;
    s_bal[threadIdx.x] = my_bal;
    __syncthreads();

    // Exclusive prefix sum
    prefix[threadIdx.x] = my_pred;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int64_t val = prefix[threadIdx.x];
        if (threadIdx.x >= offset) {
            val += prefix[threadIdx.x - offset];
        }
        __syncthreads();
        prefix[threadIdx.x] = val;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_offset = atomicAdd((unsigned long long*)out_n, (unsigned long long)prefix[blockDim.x - 1]);
    }
    __syncthreads();

    if (my_pred) {
        int64_t write_pos = (prefix[threadIdx.x] - 1) + block_offset;
        out_code[write_pos] = my_code;
        out_bal[write_pos] = my_bal;
    }
}

std::vector<torch::Tensor> q22_launcher(
    torch::Tensor custkey,
    torch::Tensor acctbal,
    torch::Tensor cntrycode,
    torch::Tensor in_mask,
    torch::Tensor sorted_okeys,
    double avg_val
) {
    auto n = custkey.size(0);
    auto okeys_size = sorted_okeys.size(0);

    auto sum_out = torch::zeros({1}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto count_out = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q22_avg_kernel<<<blocks, BLOCK_SIZE>>>(
        acctbal.data_ptr<double>(),
        in_mask.data_ptr<bool>(),
        sum_out.data_ptr<double>(),
        (long long*)count_out.data_ptr<int64_t>(),
        n
    );

    double avg = sum_out.item<double>() / (double)count_out.item<int64_t>();

    // Allocate output buffers (max size = n)
    auto out_code = torch::empty({n}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto out_bal = torch::empty({n}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto out_n = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));

    q22_filter_kernel<<<blocks, BLOCK_SIZE>>>(
        custkey.data_ptr<int64_t>(),
        acctbal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        in_mask.data_ptr<bool>(),
        sorted_okeys.data_ptr<int64_t>(),
        okeys_size,
        avg,
        out_code.data_ptr<int32_t>(),
        out_bal.data_ptr<double>(),
        (long long*)out_n.data_ptr<int64_t>(),
        n
    );

    return {out_code, out_bal, out_n};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q22_launcher", &q22_launcher, "TPC-H Q22 CUDA launcher");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q22_cuda_")
    cu_path = os.path.join(tmpdir, "_kernels.cu")
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

def query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated core for TPC-H Q22.
    Combines avg computation and filter+project into fused kernels.
    
    Args:
        c_custkey: int64 tensor of customer keys
        c_acctbal: float64 tensor of account balances
        cntrycode: int32 tensor of country codes
        in_cntrycode_mask: bool tensor mask for target country codes
        o_custkey_unique: int64 tensor of unique order customer keys
    
    Returns:
        Tuple of (cntrycode, numcust, totacctbal) tensors, sorted by cntrycode
    """
    # Move to CUDA if not already
    c_custkey = c_custkey.contiguous().cuda()
    c_acctbal = c_acctbal.contiguous().cuda()
    cntrycode = cntrycode.contiguous().cuda()
    in_cntrycode_mask = in_cntrycode_mask.contiguous().cuda()
    
    # Sort order customer keys for binary search
    sorted_okeys, _ = torch.sort(o_custkey_unique.contiguous().cuda())
    
    # Launch fused kernels
    out_code, out_bal, out_n = _EXT.q22_launcher(
        c_custkey, c_acctbal, cntrycode, in_cntrycode_mask, sorted_okeys, 0.0
    )
    
    n_filtered = out_n.item()
    out_code = out_code[:n_filtered]
    out_bal = out_bal[:n_filtered]
    
    # Group by on compacted data
    unique_codes, inverse = torch.unique(out_code, return_inverse=True)
    numcust = torch.zeros(unique_codes.numel(), dtype=torch.int64, device='cuda')
    totacctbal = torch.zeros(unique_codes.numel(), dtype=torch.float64, device='cuda')
    
    numcust.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
    totacctbal.scatter_add_(0, inverse, out_bal)
    
    # Sort by country code
    order = torch.argsort(unique_codes)
    return unique_codes[order], numcust[order], totacctbal[order]

def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: Tuple[str, ...] = ('13', '31', '23', '29', '30', '18', '17'),
) -> cudf.DataFrame:
    """
    SQL: TPC-H Q22 - Global Sales Opportunity
    
    Description: Uses fused CUDA kernels for the hot path (filter + aggregate)
    to minimize HBM traffic and cuDF round-trips. Country code derivation and
    uniqueness computation remain in cuDF for string processing.
    
    Args:
        tables: dict containing 'customer' and 'orders' DataFrames
        cntrycodes: target 2-digit country codes
    
    Returns:
        DataFrame with columns cntrycode, numcust, totacctbal
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    
    # Derive country code from phone number
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")
    
    # Build in-set mask
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)
    
    # Unique order customer keys
    o_custkey_unique = df_orders["o_custkey"].unique()
    
    # Convert to PyTorch tensors
    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)
    o_custkey_t = torch.as_tensor(o_custkey_unique.to_cupy(), dtype=torch.int64)
    
    codes_t, numcust_t, totacctbal_t = query_core(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t, o_custkey_t
    )
    
    # Format back to 2-char zero-padded strings
    codes_cudf = cudf.Series(codes_t.cpu()).astype("int32").astype("str").str.zfill(2)
    
    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_t.cpu()),
        "totacctbal": cudf.Series(totacctbal_t.cpu()),
    })
