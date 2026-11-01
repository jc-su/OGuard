import os
import tempfile
from typing import Any

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 256

__device__ __forceinline__ int64_t q15_find_supplier_index(
    int64_t supp,
    const int64_t* __restrict__ s_suppkey,
    int64_t n_sup
) {
    int64_t direct = supp - 1;
    if (direct >= 0 && direct < n_sup && s_suppkey[direct] == supp) {
        return direct;
    }

    int64_t lo = 0;
    int64_t hi = n_sup;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        int64_t v = s_suppkey[mid];
        if (v < supp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < n_sup && s_suppkey[lo] == supp) {
        return lo;
    }
    return -1;
}

__global__ void q15_aggregate_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int64_t* __restrict__ s_suppkey,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ totals,
    uint8_t* __restrict__ present,
    int64_t n_line,
    int64_t n_sup
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_line) {
        return;
    }

    int32_t sd = l_shipdate_days[i];
    if (sd >= shipdate_lo_days && sd < shipdate_hi_days) {
        int64_t supp = l_suppkey[i];
        int64_t out_idx = q15_find_supplier_index(supp, s_suppkey, n_sup);
        if (out_idx >= 0) {
            double revenue = l_extendedprice[i] * (1.0 - l_discount[i]);
            atomicAdd(totals + out_idx, revenue);
            present[out_idx] = static_cast<uint8_t>(1);
        }
    }
}

__global__ void q15_max_present_kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    double* __restrict__ partial,
    int64_t n
) {
    __shared__ double sh[BLOCK_SIZE];

    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double v = -CUDART_INF;
    if (i < n && present[i] != 0) {
        v = totals[i];
    }
    sh[threadIdx.x] = v;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            double other = sh[threadIdx.x + s];
            if (other > sh[threadIdx.x]) {
                sh[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sh[0];
    }
}

__global__ void q15_max_plain_kernel(
    const double* __restrict__ values,
    double* __restrict__ partial,
    int64_t n
) {
    __shared__ double sh[BLOCK_SIZE];

    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double v = (i < n) ? values[i] : -CUDART_INF;
    sh[threadIdx.x] = v;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            double other = sh[threadIdx.x + s];
            if (other > sh[threadIdx.x]) {
                sh[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sh[0];
    }
}

__global__ void q15_count_selected_kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    const double* __restrict__ max_value,
    unsigned long long* __restrict__ count,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && present[i] != 0 && totals[i] == max_value[0]) {
        atomicAdd(count, 1ULL);
    }
}

__global__ void q15_fill_selected_kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    const int64_t* __restrict__ s_suppkey,
    const double* __restrict__ max_value,
    unsigned long long* __restrict__ count,
    int64_t* __restrict__ sel_idx,
    int64_t* __restrict__ sel_keys,
    double* __restrict__ sel_rev,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && present[i] != 0 && totals[i] == max_value[0]) {
        unsigned long long pos = atomicAdd(count, 1ULL);
        sel_idx[pos] = i;
        sel_keys[pos] = s_suppkey[i];
        sel_rev[pos] = totals[i];
    }
}

std::vector<torch::Tensor> q15_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    torch::Tensor s_suppkey,
    int64_t shipdate_lo_days,
    int64_t shipdate_hi_days
) {
    if (!l_suppkey.is_cuda()) l_suppkey = l_suppkey.cuda();
    if (!l_extendedprice.is_cuda()) l_extendedprice = l_extendedprice.cuda();
    if (!l_discount.is_cuda()) l_discount = l_discount.cuda();
    if (!l_shipdate_days.is_cuda()) l_shipdate_days = l_shipdate_days.cuda();
    if (!s_suppkey.is_cuda()) s_suppkey = s_suppkey.cuda();

    l_suppkey = l_suppkey.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();
    s_suppkey = s_suppkey.contiguous();

    TORCH_CHECK(l_suppkey.scalar_type() == torch::kInt64, "l_suppkey must be int64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32, "l_shipdate_days must be int32 epoch days");
    TORCH_CHECK(s_suppkey.scalar_type() == torch::kInt64, "s_suppkey must be int64");
    TORCH_CHECK(l_extendedprice.numel() == l_suppkey.numel(), "lineitem tensors must have equal length");
    TORCH_CHECK(l_discount.numel() == l_suppkey.numel(), "lineitem tensors must have equal length");
    TORCH_CHECK(l_shipdate_days.numel() == l_suppkey.numel(), "lineitem tensors must have equal length");

    const c10::cuda::CUDAGuard device_guard(l_extendedprice.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t n_line = l_suppkey.numel();
    int64_t n_sup = s_suppkey.numel();

    auto double_opts = l_extendedprice.options().dtype(torch::kFloat64);
    auto byte_opts = l_extendedprice.options().dtype(torch::kUInt8);
    auto int64_opts = s_suppkey.options().dtype(torch::kInt64);

    torch::Tensor totals = torch::zeros({n_sup}, double_opts);
    torch::Tensor present = torch::zeros({n_sup}, byte_opts);

    if (n_line > 0 && n_sup > 0) {
        int blocks_line = static_cast<int>((n_line + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q15_aggregate_kernel<<<blocks_line, BLOCK_SIZE, 0, stream>>>(
            l_suppkey.data_ptr<int64_t>(),
            l_extendedprice.data_ptr<double>(),
            l_discount.data_ptr<double>(),
            l_shipdate_days.data_ptr<int32_t>(),
            s_suppkey.data_ptr<int64_t>(),
            static_cast<int32_t>(shipdate_lo_days),
            static_cast<int32_t>(shipdate_hi_days),
            totals.data_ptr<double>(),
            present.data_ptr<uint8_t>(),
            n_line,
            n_sup
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }

    int64_t max_blocks = (n_sup + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (max_blocks < 1) {
        max_blocks = 1;
    }

    torch::Tensor current = torch::empty({max_blocks}, double_opts);
    q15_max_present_kernel<<<static_cast<int>(max_blocks), BLOCK_SIZE, 0, stream>>>(
        totals.data_ptr<double>(),
        present.data_ptr<uint8_t>(),
        current.data_ptr<double>(),
        n_sup
    );
    AT_CUDA_CHECK(cudaGetLastError());

    int64_t current_n = max_blocks;
    while (current_n > 1) {
        int64_t next_n = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        torch::Tensor next = torch::empty({next_n}, double_opts);
        q15_max_plain_kernel<<<static_cast<int>(next_n), BLOCK_SIZE, 0, stream>>>(
            current.data_ptr<double>(),
            next.data_ptr<double>(),
            current_n
        );
        AT_CUDA_CHECK(cudaGetLastError());
        current = next;
        current_n = next_n;
    }

    torch::Tensor count = torch::zeros({1}, int64_opts);
    int blocks_sup = static_cast<int>((n_sup + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (blocks_sup > 0) {
        q15_count_selected_kernel<<<blocks_sup, BLOCK_SIZE, 0, stream>>>(
            totals.data_ptr<double>(),
            present.data_ptr<uint8_t>(),
            current.data_ptr<double>(),
            reinterpret_cast<unsigned long long*>(count.data_ptr<int64_t>()),
            n_sup
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }

    int64_t h_count = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(
        &h_count,
        count.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    torch::Tensor sel_idx = torch::empty({h_count}, int64_opts);
    torch::Tensor sel_keys = torch::empty({h_count}, int64_opts);
    torch::Tensor sel_rev = torch::empty({h_count}, double_opts);

    if (h_count > 0 && blocks_sup > 0) {
        torch::Tensor fill_count = torch::zeros({1}, int64_opts);
        q15_fill_selected_kernel<<<blocks_sup, BLOCK_SIZE, 0, stream>>>(
            totals.data_ptr<double>(),
            present.data_ptr<uint8_t>(),
            s_suppkey.data_ptr<int64_t>(),
            current.data_ptr<double>(),
            reinterpret_cast<unsigned long long*>(fill_count.data_ptr<int64_t>()),
            sel_idx.data_ptr<int64_t>(),
            sel_keys.data_ptr<int64_t>(),
            sel_rev.data_ptr<double>(),
            n_sup
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }

    return {sel_idx, sel_keys, sel_rev};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", &q15_launcher, "TPC-H Q15 fused revenue aggregation and top supplier selection (CUDA)");
}
'''


def _load_cuda() -> Any:
    """
    Compile the CUDA extension from the in-module _CUDA_SOURCE string.

    Runs on host at import time via torch.utils.cpp_extension.load().  The hot
    path itself is handwritten CUDA: a fused date-filter/project/group scan over
    lineitem plus GPU reductions over supplier totals.  This avoids the eager
    Torch plan's materialized boolean masks, filtered temporary arrays, sort,
    searchsorted, and scatter passes over the large lineitem table.
    """
    tmpdir = tempfile.mkdtemp(prefix="q15_cuda_")
    cu_path = os.path.join(tmpdir, "_q15_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q15_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _query_core(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU numeric core for TPC-H Q15.

    The large memory-bound lineitem path runs in CUDA: one coalesced pass reads
    shipdate first, conditionally loads supplier/price/discount only for rows in
    the date window, computes revenue in registers, and atomically accumulates
    fp64 totals by supplier.  CUDA kernels then reduce supplier totals to the
    maximum and compact the winning supplier rows.  A tiny PyTorch argsort is
    used only on the final result cardinality to enforce ORDER BY s_suppkey.
    """
    sel_idx_t, sel_keys_t, sel_rev_t = _EXT.q15_launcher(
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        s_suppkey,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    if sel_keys_t.numel() > 1:
        order = torch.argsort(sel_keys_t)
        sel_idx_t = sel_idx_t[order]
        sel_keys_t = sel_keys_t[order]
        sel_rev_t = sel_rev_t[order]

    return sel_idx_t, sel_keys_t, sel_rev_t


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1996-01-01",
    shipdate_hi: str = "1996-04-01",
) -> cudf.DataFrame:
    """
    SQL:
      WITH revenue AS (
        SELECT l_suppkey AS supplier_no,
               SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
        GROUP BY l_suppkey
      )
      SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
      FROM supplier, revenue
      WHERE s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
      ORDER BY s_suppkey;
    Description: TPC-H Q15 — find supplier(s) with the top revenue in a given quarter.
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_price_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_disc_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)

    sel_idx_t, sel_keys_t, sel_rev_t = _query_core(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        s_suppkey_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # Use indices to gather string/phone columns from cuDF supplier table
    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_t),
    })
