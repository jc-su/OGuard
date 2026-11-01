import hashlib
import os
import tempfile
from typing import Any

import cudf
import pandas as pd
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#define BLOCK_SIZE 1024
#define MAX_PRIORITIES 256

__global__ void q4_mark_late_orders_kernel(
    const int64_t* __restrict__ l_orderkey,
    const int32_t* __restrict__ l_commitdate_days,
    const int32_t* __restrict__ l_receiptdate_days,
    uint8_t* __restrict__ late_bitmap,
    int64_t n_lineitem,
    int64_t bitmap_size
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_lineitem) {
        int32_t commit_day = l_commitdate_days[idx];
        int32_t receipt_day = l_receiptdate_days[idx];
        if (commit_day < receipt_day) {
            int64_t key = l_orderkey[idx];
            if (key >= 0 && key < bitmap_size) {
                late_bitmap[key] = static_cast<uint8_t>(1);
            }
        }
    }
}

__global__ void q4_count_orders_kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_orderdate_days,
    const int32_t* __restrict__ o_priority_codes,
    const uint8_t* __restrict__ late_bitmap,
    int64_t* __restrict__ counts,
    int64_t n_orders,
    int64_t bitmap_size,
    int32_t date_lo_days,
    int32_t date_hi_days,
    int32_t num_priorities
) {
    __shared__ unsigned long long shared_counts[MAX_PRIORITIES];

    for (int i = threadIdx.x; i < MAX_PRIORITIES; i += blockDim.x) {
        shared_counts[i] = 0ULL;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int32_t od = o_orderdate_days[idx];
        if (od >= date_lo_days && od < date_hi_days) {
            int64_t key = o_orderkey[idx];
            if (key >= 0 && key < bitmap_size && late_bitmap[key] != 0) {
                int32_t code = o_priority_codes[idx];
                if (code >= 0 && code < num_priorities) {
                    atomicAdd(&shared_counts[code], 1ULL);
                }
            }
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_priorities; i += blockDim.x) {
        unsigned long long v = shared_counts[i];
        if (v != 0ULL) {
            atomicAdd(reinterpret_cast<unsigned long long*>(counts) + i, v);
        }
    }
}

torch::Tensor q4_launcher(
    torch::Tensor o_orderkey,
    torch::Tensor o_orderdate_days,
    torch::Tensor o_priority_codes,
    torch::Tensor l_orderkey,
    torch::Tensor l_commitdate_days,
    torch::Tensor l_receiptdate_days,
    int32_t date_lo_days,
    int32_t date_hi_days,
    int32_t num_priorities,
    int64_t max_lineitem_orderkey
) {
    TORCH_CHECK(o_orderkey.is_cuda(), "o_orderkey must be a CUDA tensor");
    TORCH_CHECK(o_orderdate_days.is_cuda(), "o_orderdate_days must be a CUDA tensor");
    TORCH_CHECK(o_priority_codes.is_cuda(), "o_priority_codes must be a CUDA tensor");
    TORCH_CHECK(l_orderkey.is_cuda(), "l_orderkey must be a CUDA tensor");
    TORCH_CHECK(l_commitdate_days.is_cuda(), "l_commitdate_days must be a CUDA tensor");
    TORCH_CHECK(l_receiptdate_days.is_cuda(), "l_receiptdate_days must be a CUDA tensor");

    TORCH_CHECK(o_orderkey.scalar_type() == torch::kInt64, "o_orderkey must be int64");
    TORCH_CHECK(l_orderkey.scalar_type() == torch::kInt64, "l_orderkey must be int64");
    TORCH_CHECK(o_orderdate_days.scalar_type() == torch::kInt32, "o_orderdate_days must be int32");
    TORCH_CHECK(l_commitdate_days.scalar_type() == torch::kInt32, "l_commitdate_days must be int32");
    TORCH_CHECK(l_receiptdate_days.scalar_type() == torch::kInt32, "l_receiptdate_days must be int32");
    TORCH_CHECK(o_priority_codes.scalar_type() == torch::kInt32, "o_priority_codes must be int32");
    TORCH_CHECK(num_priorities >= 0 && num_priorities <= MAX_PRIORITIES, "num_priorities exceeds MAX_PRIORITIES");
    TORCH_CHECK(max_lineitem_orderkey >= 0, "max_lineitem_orderkey must be non-negative");

    o_orderkey = o_orderkey.contiguous();
    o_orderdate_days = o_orderdate_days.contiguous();
    o_priority_codes = o_priority_codes.contiguous();
    l_orderkey = l_orderkey.contiguous();
    l_commitdate_days = l_commitdate_days.contiguous();
    l_receiptdate_days = l_receiptdate_days.contiguous();

    int64_t n_orders = o_orderkey.numel();
    int64_t n_lineitem = l_orderkey.numel();

    TORCH_CHECK(o_orderdate_days.numel() == n_orders, "orders date length mismatch");
    TORCH_CHECK(o_priority_codes.numel() == n_orders, "orders priority length mismatch");
    TORCH_CHECK(l_commitdate_days.numel() == n_lineitem, "lineitem commitdate length mismatch");
    TORCH_CHECK(l_receiptdate_days.numel() == n_lineitem, "lineitem receiptdate length mismatch");

    int64_t bitmap_size = max_lineitem_orderkey + 1;

    auto bitmap_opts = torch::TensorOptions().device(o_orderkey.device()).dtype(torch::kUInt8);
    auto count_opts = torch::TensorOptions().device(o_orderkey.device()).dtype(torch::kInt64);

    torch::Tensor late_bitmap = torch::zeros({bitmap_size}, bitmap_opts);
    torch::Tensor counts = torch::zeros({num_priorities}, count_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n_lineitem > 0) {
        int blocks_line = static_cast<int>((n_lineitem + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q4_mark_late_orders_kernel<<<blocks_line, BLOCK_SIZE, 0, stream>>>(
            l_orderkey.data_ptr<int64_t>(),
            l_commitdate_days.data_ptr<int32_t>(),
            l_receiptdate_days.data_ptr<int32_t>(),
            late_bitmap.data_ptr<uint8_t>(),
            n_lineitem,
            bitmap_size
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (n_orders > 0 && num_priorities > 0) {
        int blocks_orders = static_cast<int>((n_orders + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q4_count_orders_kernel<<<blocks_orders, BLOCK_SIZE, 0, stream>>>(
            o_orderkey.data_ptr<int64_t>(),
            o_orderdate_days.data_ptr<int32_t>(),
            o_priority_codes.data_ptr<int32_t>(),
            late_bitmap.data_ptr<uint8_t>(),
            counts.data_ptr<int64_t>(),
            n_orders,
            bitmap_size,
            date_lo_days,
            date_hi_days,
            num_priorities
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q4_launcher", &q4_launcher, "TPC-H Q4 late-order priority counts (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile and load the hand-written CUDA extension from the embedded .cu source."""
    digest = hashlib.md5(_CUDA_SOURCE.encode("utf-8")).hexdigest()[:12]
    name = f"q4_cuda_ext_sm90_{digest}"
    root = os.path.join(tempfile.gettempdir(), name)
    build_dir = os.path.join(root, "build")
    cu_path = os.path.join(root, "_q4_kernels.cu")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name=name,
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _days_tensor_from_datetime_seconds(col: cudf.Series) -> torch.Tensor:
    """
    Convert a cuDF datetime64[s] column to an int32 CUDA tensor of epoch days.

    cuDF performs the cast and integer division on GPU, and Torch wraps the resulting
    CuPy buffer without a host copy.
    """
    return torch.as_tensor(((col.astype("int64") // 86400).astype("int32")).to_cupy(), dtype=torch.int32)


def _query_core(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
    max_lineitem_orderkey: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Execute the Q4 hot path in CUDA.

    The CUDA extension builds a byte bitmap of orderkeys having at least one late
    lineitem, then scans orders once to apply the date predicate, probe the bitmap,
    and aggregate by priority code. This replaces Torch unique/isin and large
    intermediate tensors with two coalesced memory-bandwidth-oriented CUDA passes.
    """
    counts = _EXT.q4_launcher(
        o_orderkey,
        o_orderdate_days,
        o_orderpriority_codes,
        l_orderkey,
        l_commitdate_days,
        l_receiptdate_days,
        int(date_lo_days),
        int(date_hi_days),
        int(num_priorities),
        int(max_lineitem_orderkey),
    )
    priority_ids = torch.arange(num_priorities, device=counts.device, dtype=torch.int64)
    nz = counts > 0
    return priority_ids[nz], counts[nz]


def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT
            o_orderpriority,
            count(*) AS order_count
        FROM
            orders
        WHERE
            o_orderdate >= CAST('1993-07-01' AS date)
            AND o_orderdate < CAST('1993-10-01' AS date)
            AND EXISTS (
                SELECT *
                FROM lineitem
                WHERE l_orderkey = o_orderkey
                  AND l_commitdate < l_receiptdate)
        GROUP BY
            o_orderpriority
        ORDER BY
            o_orderpriority;

    Description:
        TPC-H Q4. cuDF handles priority string categorical encoding, datetime-to-day
        conversion, and final small result construction/sort. The bulk filter/EXISTS/
        aggregation path runs in hand-written CUDA: a lineitem scan marks late
        orderkeys in a GPU bitmap, and an orders scan filters by date, probes the
        bitmap, and counts by priority with per-block shared counters. This avoids
        the reference Torch unique/isin materialization and large filtered temporaries.

    Params:
        tables (dict[str, cudf.DataFrame]): Input tables containing "orders" and
            "lineitem".
        date_lo (str): Inclusive lower bound for o_orderdate.
        date_hi (str): Exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)
    o_orderdate_days_t = _days_tensor_from_datetime_seconds(df_orders["o_orderdate"])
    o_priority_codes_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32)

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_commitdate_days_t = _days_tensor_from_datetime_seconds(df_lineitem["l_commitdate"])
    l_receiptdate_days_t = _days_tensor_from_datetime_seconds(df_lineitem["l_receiptdate"])

    max_lineitem_orderkey = int(df_lineitem["l_orderkey"].max()) if len(df_lineitem) else 0

    priority_ids_t, counts_t = _query_core(
        o_orderkey_t,
        o_orderdate_days_t,
        o_priority_codes_t,
        l_orderkey_t,
        l_commitdate_days_t,
        l_receiptdate_days_t,
        date_lo_days,
        date_hi_days,
        num_priorities,
        max_lineitem_orderkey,
    )

    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[int(i)] for i in priority_ids_host]

    result = cudf.DataFrame(
        {
            "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
            "order_count": counts_t,
        }
    )
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
