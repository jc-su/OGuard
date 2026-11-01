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

#define BLOCK_SIZE 1024

__global__ void q22_avg_kernel(
    const double* __restrict__ acctbal_ptr,
    const int32_t* __restrict__ cntrycode_mask_ptr,
    double* __restrict__ sum_out,
    int64_t* __restrict__ count_out,
    int64_t n
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ int64_t block_count[BLOCK_SIZE];
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double my_sum = 0.0;
    int64_t my_count = 0;
    if (idx < n) {
        if (cntrycode_mask_ptr[idx] && acctbal_ptr[idx] > 0.0) {
            my_sum = acctbal_ptr[idx];
            my_count = 1;
        }
    }
    block_sum[threadIdx.x] = my_sum;
    block_count[threadIdx.x] = my_count;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            block_count[threadIdx.x] += block_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum[0]);
        atomicAdd((unsigned long long*)count_out, (unsigned long long)block_count[0]);
    }
}

__global__ void q22_filter_kernel(
    const int64_t* __restrict__ custkey_ptr,
    const double* __restrict__ acctbal_ptr,
    const int32_t* __restrict__ cntrycode_ptr,
    const int32_t* __restrict__ in_cntrycode_mask_ptr,
    const int64_t* __restrict__ sorted_okeys_ptr,
    int64_t n_okeys,
    double avg_val,
    int32_t* __restrict__ out_cntrycodes,
    double* __restrict__ out_acctbals,
    int32_t* __restrict__ out_count,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n) return;

    if (!in_cntrycode_mask_ptr[idx]) return;

    double bal = acctbal_ptr[idx];
    if (!(bal > avg_val)) return;

    int64_t ck = custkey_ptr[idx];

    // Binary search sorted_okeys
    int64_t lo = 0, hi = n_okeys;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (sorted_okeys_ptr[mid] < ck) lo = mid + 1;
        else hi = mid;
    }

    if (lo < n_okeys && sorted_okeys_ptr[lo] == ck) return;

    // Atomically write to output buffer
    int32_t pos = atomicAdd(out_count, 1);
    if (pos < n) {
        out_cntrycodes[pos] = cntrycode_ptr[idx];
        out_acctbals[pos] = bal;
    }
}

__global__ void q22_groupby_kernel(
    const int32_t* __restrict__ codes,
    const double* __restrict__ bals,
    int32_t n_filtered,
    const int32_t* __restrict__ unique_codes,
    int32_t* __restrict__ numcust_out,
    double* __restrict__ totacctbal_out,
    int32_t n_unique
) {
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n_filtered) return;

    int32_t cc = codes[idx];
    double bal = bals[idx];

    // Linear search for small group-by
    for (int32_t i = 0; i < n_unique; i++) {
        if (unique_codes[i] == cc) {
            atomicAdd(&numcust_out[i], 1);
            atomicAdd(&totacctbal_out[i], bal);
            return;
        }
    }
}

void q22_avg_launcher(
    torch::Tensor acctbal,
    torch::Tensor cntrycode_mask,
    torch::Tensor sum_out,
    torch::Tensor count_out
) {
    auto n = acctbal.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q22_avg_kernel<<<blocks, BLOCK_SIZE>>>(
        acctbal.data_ptr<double>(),
        cntrycode_mask.data_ptr<int32_t>(),
        sum_out.data_ptr<double>(),
        count_out.data_ptr<int64_t>(),
        n
    );
}

void q22_filter_launcher(
    torch::Tensor custkey,
    torch::Tensor acctbal,
    torch::Tensor cntrycode,
    torch::Tensor in_cntrycode_mask,
    torch::Tensor sorted_okeys,
    double avg_val,
    torch::Tensor out_cntrycodes,
    torch::Tensor out_acctbals,
    torch::Tensor out_count
) {
    auto n = custkey.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q22_filter_kernel<<<blocks, BLOCK_SIZE>>>(
        custkey.data_ptr<int64_t>(),
        acctbal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        in_cntrycode_mask.data_ptr<int32_t>(),
        sorted_okeys.data_ptr<int64_t>(),
        sorted_okeys.numel(),
        avg_val,
        out_cntrycodes.data_ptr<int32_t>(),
        out_acctbals.data_ptr<double>(),
        out_count.data_ptr<int32_t>(),
        n
    );
}

void q22_groupby_launcher(
    torch::Tensor codes,
    torch::Tensor bals,
    int32_t n_filtered,
    torch::Tensor unique_codes,
    torch::Tensor numcust_out,
    torch::Tensor totacctbal_out,
    int32_t n_unique
) {
    int blocks = (n_filtered + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q22_groupby_kernel<<<blocks, BLOCK_SIZE>>>(
        codes.data_ptr<int32_t>(),
        bals.data_ptr<double>(),
        n_filtered,
        unique_codes.data_ptr<int32_t>(),
        numcust_out.data_ptr<int32_t>(),
        totacctbal_out.data_ptr<double>(),
        n_unique
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q22_avg_launcher", &q22_avg_launcher, "TPC-H Q22 average (CUDA)");
    m.def("q22_filter_launcher", &q22_filter_launcher, "TPC-H Q22 filter (CUDA)");
    m.def("q22_groupby_launcher", &q22_groupby_launcher, "TPC-H Q22 groupby (CUDA)");
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
    CUDA-accelerated TPC-H Q22 core.
    Uses fused CUDA kernels for avg computation, filter+exists, and groupby.
    Memory-bound; fuses multiple passes into fewer kernel launches.
    """
    device = c_custkey.device

    # Step 1: avg(c_acctbal) where c_acctbal > 0 AND cntrycode in set
    avg_sum = torch.zeros(1, dtype=torch.float64, device=device)
    avg_count = torch.zeros(1, dtype=torch.int64, device=device)
    _EXT.q22_avg_launcher(c_acctbal, in_cntrycode_mask.to(torch.int32), avg_sum, avg_count)
    avg_val = (avg_sum.item() / avg_count.item()) if avg_count.item() > 0 else 0.0

    # Step 2: sort o_custkey_unique for binary search
    sorted_okeys, _ = torch.sort(o_custkey_unique)

    # Step 3: filter and output filtered data
    n = c_custkey.numel()
    out_codes = torch.zeros(n, dtype=torch.int32, device=device)
    out_bals = torch.zeros(n, dtype=torch.float64, device=device)
    out_count = torch.zeros(1, dtype=torch.int32, device=device)
    _EXT.q22_filter_launcher(
        c_custkey, c_acctbal, cntrycode, in_cntrycode_mask.to(torch.int32),
        sorted_okeys, avg_val, out_codes, out_bals, out_count
    )
    n_filtered = out_count.item()
    if n_filtered == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device)
        )

    f_codes = out_codes[:n_filtered]
    f_bals = out_bals[:n_filtered]

    # Step 4: group by cntrycode
    unique_codes = torch.unique(f_codes)
    n_unique = unique_codes.numel()
    numcust = torch.zeros(n_unique, dtype=torch.int32, device=device)
    totacctbal = torch.zeros(n_unique, dtype=torch.float64, device=device)
    _EXT.q22_groupby_launcher(f_codes, f_bals, n_filtered, unique_codes, numcust, totacctbal, n_unique)

    # Step 5: order by cntrycode
    order = torch.argsort(unique_codes)
    return unique_codes[order], numcust[order].to(torch.int64), totacctbal[order]


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
