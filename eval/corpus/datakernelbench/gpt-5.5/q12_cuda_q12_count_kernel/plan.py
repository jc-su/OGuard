import os
import tempfile
from types import ModuleType
from typing import Any

import cudf
import pandas as pd
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(val));
}

__global__ void q12_count_kernel(
    const int32_t* __restrict__ shipmode_code,
    const int32_t* __restrict__ priority_high,
    const int32_t* __restrict__ commit_days,
    const int32_t* __restrict__ receipt_days,
    const int32_t* __restrict__ ship_days,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days,
    int64_t* __restrict__ out_counts,
    int64_t n_elements
) {
    __shared__ int64_t sh_high0[BLOCK_SIZE];
    __shared__ int64_t sh_high1[BLOCK_SIZE];
    __shared__ int64_t sh_low0[BLOCK_SIZE];
    __shared__ int64_t sh_low1[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int64_t high0 = 0;
    int64_t high1 = 0;
    int64_t low0 = 0;
    int64_t low1 = 0;

    if (idx < n_elements) {
        const int32_t cd = commit_days[idx];
        const int32_t rd = receipt_days[idx];
        const int32_t sd = ship_days[idx];

        const bool keep = (cd < rd) &&
                          (sd < cd) &&
                          (rd >= receipt_lo_days) &&
                          (rd < receipt_hi_days);

        if (keep) {
            const int32_t mode = shipmode_code[idx];
            const int32_t high = priority_high[idx];

            if (mode == 0) {
                if (high != 0) {
                    high0 = 1;
                } else {
                    low0 = 1;
                }
            } else {
                if (high != 0) {
                    high1 = 1;
                } else {
                    low1 = 1;
                }
            }
        }
    }

    sh_high0[tid] = high0;
    sh_high1[tid] = high1;
    sh_low0[tid] = low0;
    sh_low1[tid] = low1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_high0[tid] += sh_high0[tid + stride];
            sh_high1[tid] += sh_high1[tid + stride];
            sh_low0[tid] += sh_low0[tid + stride];
            sh_low1[tid] += sh_low1[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_add_i64(out_counts + 0, sh_high0[0]);
        atomic_add_i64(out_counts + 1, sh_high1[0]);
        atomic_add_i64(out_counts + 2, sh_low0[0]);
        atomic_add_i64(out_counts + 3, sh_low1[0]);
    }
}

torch::Tensor q12_launcher(
    torch::Tensor shipmode_code,
    torch::Tensor priority_high,
    torch::Tensor commit_days,
    torch::Tensor receipt_days,
    torch::Tensor ship_days,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days
) {
    if (!shipmode_code.is_cuda()) {
        shipmode_code = shipmode_code.to(torch::kCUDA);
    }
    if (!priority_high.is_cuda()) {
        priority_high = priority_high.to(torch::kCUDA);
    }
    if (!commit_days.is_cuda()) {
        commit_days = commit_days.to(torch::kCUDA);
    }
    if (!receipt_days.is_cuda()) {
        receipt_days = receipt_days.to(torch::kCUDA);
    }
    if (!ship_days.is_cuda()) {
        ship_days = ship_days.to(torch::kCUDA);
    }

    TORCH_CHECK(shipmode_code.scalar_type() == torch::kInt32, "shipmode_code must be int32");
    TORCH_CHECK(priority_high.scalar_type() == torch::kInt32, "priority_high must be int32");
    TORCH_CHECK(commit_days.scalar_type() == torch::kInt32, "commit_days must be int32");
    TORCH_CHECK(receipt_days.scalar_type() == torch::kInt32, "receipt_days must be int32");
    TORCH_CHECK(ship_days.scalar_type() == torch::kInt32, "ship_days must be int32");

    shipmode_code = shipmode_code.contiguous();
    priority_high = priority_high.contiguous();
    commit_days = commit_days.contiguous();
    receipt_days = receipt_days.contiguous();
    ship_days = ship_days.contiguous();

    const int64_t n = shipmode_code.numel();
    TORCH_CHECK(priority_high.numel() == n, "priority_high length mismatch");
    TORCH_CHECK(commit_days.numel() == n, "commit_days length mismatch");
    TORCH_CHECK(receipt_days.numel() == n, "receipt_days length mismatch");
    TORCH_CHECK(ship_days.numel() == n, "ship_days length mismatch");

    const c10::cuda::CUDAGuard device_guard(shipmode_code.device());
    auto out = torch::zeros({4}, torch::TensorOptions()
                                     .dtype(torch::kInt64)
                                     .device(shipmode_code.device()));

    if (n > 0) {
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q12_count_kernel<<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            shipmode_code.data_ptr<int32_t>(),
            priority_high.data_ptr<int32_t>(),
            commit_days.data_ptr<int32_t>(),
            receipt_days.data_ptr<int32_t>(),
            ship_days.data_ptr<int32_t>(),
            receipt_lo_days,
            receipt_hi_days,
            out.data_ptr<int64_t>(),
            n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher, "TPC-H Q12 fused date filter and grouped counts (CUDA)");
}
'''


def _load_cuda() -> ModuleType:
    """Compile the CUDA extension from the embedded .cu source at import time."""
    tmpdir = tempfile.mkdtemp(prefix="q12_cuda_")
    cu_path = os.path.join(tmpdir, "_q12_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q12_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT: ModuleType = _load_cuda()


def _as_int32_torch(series: cudf.Series) -> torch.Tensor:
    """Convert a cuDF int32 Series to a CUDA torch tensor without CPU materialization."""
    return torch.as_tensor(series.to_cupy(), dtype=torch.int32)


def _date_days_i32(series: cudf.Series) -> cudf.Series:
    """Convert cuDF datetime64[s] to int32 epoch days on GPU using cuDF arithmetic."""
    return ((series.astype("int64") // 86400).astype("int32"))


def _query_core(
    l_shipmode_code: torch.Tensor,
    o_orderpriority_high: torch.Tensor,
    l_commitdate: torch.Tensor,
    l_receiptdate: torch.Tensor,
    l_shipdate: torch.Tensor,
    receipt_lo_days: int,
    receipt_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the hot path in a hand-written CUDA kernel.

    The CUDA kernel performs the Q12 date predicates and the two fixed shipmode
    grouped high/low priority counts in one coalesced scan. This avoids PyTorch
    boolean-mask temporaries, masked gathers, and scatter_add kernels from the
    reference plan.
    """
    counts: torch.Tensor = _EXT.q12_launcher(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        int(receipt_lo_days),
        int(receipt_hi_days),
    )
    group_ids = torch.arange(2, device=counts.device, dtype=torch.int64)
    high_counts = counts[0:2]
    low_counts = counts[2:4]
    return group_ids, high_counts, low_counts


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipmode1: str = "MAIL",
    shipmode2: str = "SHIP",
    receipt_lo: str = "1994-01-01",
    receipt_hi: str = "1995-01-01",
    priority1: str = "1-URGENT",
    priority2: str = "2-HIGH",
) -> cudf.DataFrame:
    """
    SQL:
      SELECT l_shipmode,
             SUM(CASE WHEN o_orderpriority = '1-URGENT'
                       OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
             SUM(CASE WHEN o_orderpriority <> '1-URGENT'
                       AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
      FROM orders, lineitem
      WHERE o_orderkey = l_orderkey
        AND l_shipmode IN ('MAIL', 'SHIP')
        AND l_commitdate < l_receiptdate
        AND l_shipdate < l_commitdate
        AND l_receiptdate >= DATE '1994-01-01'
        AND l_receiptdate < DATE '1995-01-01'
      GROUP BY l_shipmode
      ORDER BY l_shipmode;

    Description:
      TPC-H Q12. cuDF handles string predicates, int-day date preparation, and
      the order/lineitem hash join using only projected columns needed by the
      query. The memory-bound post-join hot path runs in one fused CUDA kernel:
      it scans int32 date/code/priority columns once, applies all date filters in
      registers, conditionally loads group/priority values, reduces counts in
      shared memory per block, and performs only one global atomic per block per
      output counter. This removes the reference plan's PyTorch mask materializa-
      tion, masked compaction, and scatter_add passes.

    Params:
      tables: Mapping containing cuDF DataFrames "orders" and "lineitem".
      shipmode1, shipmode2: Shipping modes included in the IN predicate.
      receipt_lo, receipt_hi: Inclusive/exclusive receipt-date range.
      priority1, priority2: Order priorities counted as high priority.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    li_mask = df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])
    li = df_lineitem[li_mask]

    li_small = cudf.DataFrame(
        {
            "l_orderkey": li["l_orderkey"],
            "_shipmode_code": (li["l_shipmode"] == shipmode2).astype("int32"),
            "_commit_days": _date_days_i32(li["l_commitdate"]),
            "_receipt_days": _date_days_i32(li["l_receiptdate"]),
            "_ship_days": _date_days_i32(li["l_shipdate"]),
        }
    )

    priority_high = (
        (df_orders["o_orderpriority"] == priority1)
        | (df_orders["o_orderpriority"] == priority2)
    ).astype("int32")

    orders_small = cudf.DataFrame(
        {
            "o_orderkey": df_orders["o_orderkey"],
            "_priority_high": priority_high,
        }
    )

    joined = li_small.merge(
        orders_small,
        left_on="l_orderkey",
        right_on="o_orderkey",
        how="inner",
    )

    shipmode_t = _as_int32_torch(joined["_shipmode_code"])
    priority_t = _as_int32_torch(joined["_priority_high"])
    commit_t = _as_int32_torch(joined["_commit_days"])
    receipt_t = _as_int32_torch(joined["_receipt_days"])
    ship_t = _as_int32_torch(joined["_ship_days"])

    _, high_counts, low_counts = _query_core(
        shipmode_t,
        priority_t,
        commit_t,
        receipt_t,
        ship_t,
        receipt_lo_days,
        receipt_hi_days,
    )

    names = [shipmode1, shipmode2]
    ordered = sorted(range(2), key=lambda i: names[i])

    hc_host = high_counts.cpu().tolist()
    lc_host = low_counts.cpu().tolist()

    return cudf.DataFrame(
        {
            "l_shipmode": [names[i] for i in ordered],
            "high_line_count": [hc_host[i] for i in ordered],
            "low_line_count": [lc_host[i] for i in ordered],
        }
    )
