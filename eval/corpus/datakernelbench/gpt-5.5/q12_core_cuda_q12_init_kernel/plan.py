import os
import tempfile
from typing import Any

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 1024

__global__ void q12_init_kernel(
    int64_t* __restrict__ group_ids,
    int64_t* __restrict__ high_counts,
    int64_t* __restrict__ low_counts
) {
    if (threadIdx.x == 0) {
        group_ids[0] = 0;
        group_ids[1] = 1;
        high_counts[0] = 0;
        high_counts[1] = 0;
        low_counts[0] = 0;
        low_counts[1] = 0;
    }
}

__device__ __forceinline__ void atomic_add_i64_nonnegative(
    int64_t* addr,
    long long value
) {
    if (value != 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(addr),
            static_cast<unsigned long long>(value)
        );
    }
}

__global__ void q12_kernel(
    const int32_t* __restrict__ l_shipmode_code,
    const int32_t* __restrict__ o_orderpriority_high,
    const int32_t* __restrict__ l_commitdate,
    const int32_t* __restrict__ l_receiptdate,
    const int32_t* __restrict__ l_shipdate,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days,
    int64_t n_elements,
    int64_t* __restrict__ high_counts,
    int64_t* __restrict__ low_counts
) {
    __shared__ long long shared_counts[4 * BLOCK_SIZE];

    long long* sh_h0 = shared_counts;
    long long* sh_h1 = shared_counts + BLOCK_SIZE;
    long long* sh_l0 = shared_counts + 2 * BLOCK_SIZE;
    long long* sh_l1 = shared_counts + 3 * BLOCK_SIZE;

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    long long h0 = 0;
    long long h1 = 0;
    long long l0 = 0;
    long long l1 = 0;

    if (idx < n_elements) {
        const int32_t commit_d = l_commitdate[idx];
        const int32_t receipt_d = l_receiptdate[idx];
        const int32_t ship_d = l_shipdate[idx];

        const bool keep =
            (commit_d < receipt_d) &&
            (ship_d < commit_d) &&
            (receipt_d >= receipt_lo_days) &&
            (receipt_d < receipt_hi_days);

        if (keep) {
            const int32_t code = l_shipmode_code[idx];
            const int32_t high = o_orderpriority_high[idx] != 0 ? 1 : 0;
            const int32_t low = 1 - high;

            if (code == 0) {
                h0 = static_cast<long long>(high);
                l0 = static_cast<long long>(low);
            } else {
                h1 = static_cast<long long>(high);
                l1 = static_cast<long long>(low);
            }
        }
    }

    sh_h0[tid] = h0;
    sh_h1[tid] = h1;
    sh_l0[tid] = l0;
    sh_l1[tid] = l1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_h0[tid] += sh_h0[tid + stride];
            sh_h1[tid] += sh_h1[tid + stride];
            sh_l0[tid] += sh_l0[tid + stride];
            sh_l1[tid] += sh_l1[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_add_i64_nonnegative(&high_counts[0], sh_h0[0]);
        atomic_add_i64_nonnegative(&high_counts[1], sh_h1[0]);
        atomic_add_i64_nonnegative(&low_counts[0], sh_l0[0]);
        atomic_add_i64_nonnegative(&low_counts[1], sh_l1[0]);
    }
}

std::vector<torch::Tensor> q12_launcher(
    torch::Tensor l_shipmode_code,
    torch::Tensor o_orderpriority_high,
    torch::Tensor l_commitdate,
    torch::Tensor l_receiptdate,
    torch::Tensor l_shipdate,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days
) {
    TORCH_CHECK(l_shipmode_code.is_cuda(), "l_shipmode_code must be a CUDA tensor");
    TORCH_CHECK(o_orderpriority_high.is_cuda(), "o_orderpriority_high must be a CUDA tensor");
    TORCH_CHECK(l_commitdate.is_cuda(), "l_commitdate must be a CUDA tensor");
    TORCH_CHECK(l_receiptdate.is_cuda(), "l_receiptdate must be a CUDA tensor");
    TORCH_CHECK(l_shipdate.is_cuda(), "l_shipdate must be a CUDA tensor");

    TORCH_CHECK(l_shipmode_code.scalar_type() == torch::kInt32, "l_shipmode_code must be int32");
    TORCH_CHECK(o_orderpriority_high.scalar_type() == torch::kInt32, "o_orderpriority_high must be int32");
    TORCH_CHECK(l_commitdate.scalar_type() == torch::kInt32, "l_commitdate must be int32 epoch days");
    TORCH_CHECK(l_receiptdate.scalar_type() == torch::kInt32, "l_receiptdate must be int32 epoch days");
    TORCH_CHECK(l_shipdate.scalar_type() == torch::kInt32, "l_shipdate must be int32 epoch days");

    const int64_t n = l_shipmode_code.numel();
    TORCH_CHECK(o_orderpriority_high.numel() == n, "o_orderpriority_high length mismatch");
    TORCH_CHECK(l_commitdate.numel() == n, "l_commitdate length mismatch");
    TORCH_CHECK(l_receiptdate.numel() == n, "l_receiptdate length mismatch");
    TORCH_CHECK(l_shipdate.numel() == n, "l_shipdate length mismatch");

    c10::cuda::CUDAGuard device_guard(l_shipmode_code.device());

    l_shipmode_code = l_shipmode_code.contiguous();
    o_orderpriority_high = o_orderpriority_high.contiguous();
    l_commitdate = l_commitdate.contiguous();
    l_receiptdate = l_receiptdate.contiguous();
    l_shipdate = l_shipdate.contiguous();

    auto out_options = l_shipmode_code.options().dtype(torch::kInt64);
    torch::Tensor group_ids = torch::empty({2}, out_options);
    torch::Tensor high_counts = torch::empty({2}, out_options);
    torch::Tensor low_counts = torch::empty({2}, out_options);

    q12_init_kernel<<<1, 32>>>(
        group_ids.data_ptr<int64_t>(),
        high_counts.data_ptr<int64_t>(),
        low_counts.data_ptr<int64_t>()
    );

    if (n > 0) {
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q12_kernel<<<blocks, BLOCK_SIZE>>>(
            l_shipmode_code.data_ptr<int32_t>(),
            o_orderpriority_high.data_ptr<int32_t>(),
            l_commitdate.data_ptr<int32_t>(),
            l_receiptdate.data_ptr<int32_t>(),
            l_shipdate.data_ptr<int32_t>(),
            receipt_lo_days,
            receipt_hi_days,
            n,
            high_counts.data_ptr<int64_t>(),
            low_counts.data_ptr<int64_t>()
        );
    }

    return {group_ids, high_counts, low_counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher, "TPC-H Q12 fused filter and grouped count CUDA launcher");
}
'''


def _load_cuda() -> Any:
    """
    Compile and load the CUDA extension from an inlined .cu source file.

    The hot path runs in hand-written CUDA: a single fused GPU scan applies the
    date predicates and performs two fixed-group count reductions. This avoids
    PyTorch eager boolean masks, temporary filtered tensors, and scatter_add
    passes, which is faster for this memory-bandwidth-bound column scan.
    """
    tmpdir = tempfile.mkdtemp(prefix="q12_cuda_")
    cu_path = os.path.join(tmpdir, "_q12_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q12_cuda_core_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )


_EXT = _load_cuda()


def _launch_q12(
    l_shipmode_code: torch.Tensor,
    o_orderpriority_high: torch.Tensor,
    l_commitdate: torch.Tensor,
    l_receiptdate: torch.Tensor,
    l_shipdate: torch.Tensor,
    receipt_lo_days: int,
    receipt_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Launch the fused CUDA implementation of the Q12 tensor core.

    Inputs are already numeric 1D CUDA tensors prepared by cuDF/PyTorch in
    run_query. CUDA performs one coalesced pass over the columns, keeps all
    predicate/projection work in registers, reduces four counters per block in
    shared memory, and emits only one global atomic per counter per block.
    """
    outputs = _EXT.q12_launcher(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        int(receipt_lo_days),
        int(receipt_hi_days),
    )
    return outputs[0], outputs[1], outputs[2]


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
    CUDA tensor hot path for TPC-H Q12 after cuDF has performed join/string prep.

    This replaces PyTorch eager mask creation, masked gathers, casts, and
    scatter_add with one fused GPU kernel. The query core is memory-bound
    because it scans a handful of int32 columns with minimal arithmetic, so
    reducing HBM traffic and temporary tensors is the main speed win.
    """
    return _launch_q12(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
    )


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
    SQL: SELECT l_shipmode, SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
         SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
         FROM orders, lineitem WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL','SHIP')
         AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
         AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01'
         GROUP BY l_shipmode ORDER BY l_shipmode;
    Description: TPC-H Q12 - counts urgent vs non-urgent orders by ship mode.
    Params:
      shipmode1, shipmode2 (str): allowed shipping modes.
      receipt_lo, receipt_hi (str): receipt date range.
      priority1, priority2 (str): high-priority order priority values.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    # Pre-filter lineitem on shipmode (reduces join size)
    li = df_lineitem[df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])]

    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey")

    # Encode shipmode: MAIL=0, SHIP=1
    shipmode_code = (joined["l_shipmode"] == shipmode2).astype("int32")
    priority_high = (
        (joined["o_orderpriority"] == priority1)
        | (joined["o_orderpriority"] == priority2)
    ).astype("int32")

    l_commit_t = torch.as_tensor(
        ((joined["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_receipt_t = torch.as_tensor(
        ((joined["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_ship_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    shipmode_t = torch.as_tensor(shipmode_code.to_cupy(), dtype=torch.int32)
    priority_t = torch.as_tensor(priority_high.to_cupy(), dtype=torch.int32)

    group_ids, high_counts, low_counts = _query_core(
        shipmode_t,
        priority_t,
        l_commit_t,
        l_receipt_t,
        l_ship_t,
        receipt_lo_days,
        receipt_hi_days,
    )

    # Decode group ids back to strings; order alphabetically (MAIL < SHIP)
    names = [shipmode1, shipmode2]  # index 0 -> MAIL, 1 -> SHIP
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_counts.cpu().tolist()
    lc_host = low_counts.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count": [lc_host[i] for i in ordered],
    })
    return result
