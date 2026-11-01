import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q22_stage1_kernel(
    const double* __restrict__ c_acctbal_ptr,
    const bool* __restrict__ in_cntrycode_mask_ptr,
    double* __restrict__ avg_placeholder_ptr,
    int64_t n_customer
) {
    extern __shared__ double sdata[];
    double* block_balance = sdata;
    double* block_count = sdata + blockDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double my_balance = 0.0;
    double my_count = 0.0;
    if (idx < n_customer) {
        bool keep = in_cntrycode_mask_ptr[idx] && (c_acctbal_ptr[idx] > 0.0);
        if (keep) {
            my_balance = c_acctbal_ptr[idx];
            my_count = 1.0;
        }
    }

    block_balance[threadIdx.x] = my_balance;
    block_count[threadIdx.x] = my_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_balance[threadIdx.x] += block_balance[threadIdx.x + s];
            block_count[threadIdx.x] += block_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(avg_placeholder_ptr, block_balance[0]);
        atomicAdd(avg_placeholder_ptr + 1, block_count[0]);
    }
}

__global__ void q22_stage2_kernel(
    const int64_t* __restrict__ c_custkey_ptr,
    const double* __restrict__ c_acctbal_ptr,
    const int32_t* __restrict__ cntrycode_ptr,
    const bool* __restrict__ in_cntrycode_mask_ptr,
    const int64_t* __restrict__ o_custkey_ptr,
    double avg_val,
    double* __restrict__ numcust_buff,
    double* __restrict__ totacctbal_buff,
    int64_t n_customer,
    int64_t n_orders
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_customer) {
        int64_t custkey = c_custkey_ptr[idx];
        double acctbal = c_acctbal_ptr[idx];
        int32_t cntry = cntrycode_ptr[idx];
        bool in_mask = in_cntrycode_mask_ptr[idx];

        int64_t left = 0, right = n_orders;
        bool exists = false;
        while (left < right) {
            int64_t mid = (left + right) / 2;
            int64_t mid_key = o_custkey_ptr[mid];
            if (mid_key < custkey) {
                left = mid + 1;
            } else if (mid_key > custkey) {
                right = mid;
            } else {
                exists = true;
                break;
            }
        }

        if (in_mask && (acctbal > avg_val) && !exists) {
            atomicAdd(numcust_buff + cntry, 1.0);
            atomicAdd(totacctbal_buff + cntry, acctbal);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q22_stage1", [](torch::Tensor c_acctbal, torch::Tensor in_cntrycode_mask, torch::Tensor avg_placeholder, int64_t n_customer) {
        c_acctbal = c_acctbal.cuda().contiguous();
        in_cntrycode_mask = in_cntrycode_mask.cuda().contiguous();
        avg_placeholder = avg_placeholder.cuda();

        int blocks = (n_customer + BLOCK_SIZE - 1) / BLOCK_SIZE;
        q22_stage1_kernel<<<blocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(double)>>>(
            c_acctbal.data_ptr<double>(),
            in_cntrycode_mask.data_ptr<bool>(),
            avg_placeholder.data_ptr<double>(),
            n_customer
        );
        cudaDeviceSynchronize();
    });

    m.def("q22_stage2", [](torch::Tensor c_custkey, torch::Tensor c_acctbal, torch::Tensor cntrycode,
                           torch::Tensor in_cntrycode_mask, torch::Tensor o_custkey,
                           double avg_val, torch::Tensor numcust_buff, torch::Tensor totacctbal_buff,
                           int64_t n_customer, int64_t n_orders) {
        c_custkey = c_custkey.cuda().contiguous();
        c_acctbal = c_acctbal.cuda().contiguous();
        cntrycode = cntrycode.cuda().contiguous();
        in_cntrycode_mask = in_cntrycode_mask.cuda().contiguous();
        o_custkey = o_custkey.cuda().contiguous();
        numcust_buff = numcust_buff.cuda();
        totacctbal_buff = totacctbal_buff.cuda();

        int blocks = (n_customer + BLOCK_SIZE - 1) / BLOCK_SIZE;
        q22_stage2_kernel<<<blocks, BLOCK_SIZE>>>(
            c_custkey.data_ptr<int64_t>(),
            c_acctbal.data_ptr<double>(),
            cntrycode.data_ptr<int32_t>(),
            in_cntrycode_mask.data_ptr<bool>(),
            o_custkey.data_ptr<int64_t>(),
            avg_val,
            numcust_buff.data_ptr<double>(),
            totacctbal_buff.data_ptr<double>(),
            n_customer, n_orders
        );
        cudaDeviceSynchronize();
    });
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
    Description: TPC-H Q22 - Global sales opportunity query. Uses two-stage CUDA pipeline:
                 stage1 computes average balance, stage2 performs filter+aggregate. Both stages
                 use efficient coalesced memory access and minimize HBM transfers.
    Params: cntrycodes (tuple[str,...]): target 2-digit country codes.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)

    o_custkey_unique_series = df_orders["o_custkey"].unique()
    o_custkey_sorted = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)
    o_custkey_sorted, _ = torch.sort(o_custkey_sorted)

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)

    n_customer = c_custkey_t.numel()
    n_orders = o_custkey_sorted.numel()

    avg_placeholder = torch.zeros(2, dtype=torch.float64, device='cuda')
    _EXT.q22_stage1(c_acctbal_t, in_mask_t, avg_placeholder, n_customer)

    avg_val = avg_placeholder[0].item() / avg_placeholder[1].item()

    numcust_buff = torch.zeros(100, dtype=torch.float64, device='cuda')
    totacctbal_buff = torch.zeros(100, dtype=torch.float64, device='cuda')

    _EXT.q22_stage2(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t,
        o_custkey_sorted,
        avg_val,
        numcust_buff, totacctbal_buff,
        n_customer, n_orders
    )

    results = []
    for c in codes_int:
        cnt = int(numcust_buff[c].item())
        tot = totacctbal_buff[c].item()
        if cnt > 0 or tot > 0.0:
            results.append((c, cnt, tot))

    results.sort(key=lambda x: x[0])

    codes_str = [f"{c:02d}" for c, _, _ in results]
    numcust = [cnt for _, cnt, _ in results]
    totacctbal = [tot for _, _, tot in results]

    return cudf.DataFrame({
        "cntrycode": cudf.Series(codes_str),
        "numcust": cudf.Series(numcust),
        "totacctbal": cudf.Series(totacctbal),
    })
