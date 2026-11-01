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
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cstdint>
#include <vector>

#define BLOCK_SIZE 1024

__global__ void q10_max_reduce_kernel(
    const int64_t* __restrict__ keys,
    int64_t* __restrict__ block_max,
    int64_t n
) {
    __shared__ int64_t smem[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    int64_t v = -9223372036854775807LL - 1LL;

    if (idx < n) {
        v = keys[idx];
    }
    smem[tid] = v;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const int64_t other = smem[tid + s];
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

__global__ void q10_final_max_kernel(
    const int64_t* __restrict__ block_max,
    int64_t* __restrict__ out,
    int64_t n_blocks
) {
    __shared__ int64_t smem[BLOCK_SIZE];

    const int tid = threadIdx.x;
    int64_t v = -9223372036854775807LL - 1LL;

    for (int64_t i = tid; i < n_blocks; i += blockDim.x) {
        const int64_t x = block_max[i];
        if (x > v) {
            v = x;
        }
    }

    smem[tid] = v;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const int64_t other = smem[tid + s];
            if (other > smem[tid]) {
                smem[tid] = other;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = smem[0];
    }
}

__global__ void q10_dense_aggregate_kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    double* __restrict__ revenue_dense,
    uint8_t* __restrict__ seen,
    int64_t n,
    int64_t dense_size
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < n) {
        const int64_t k = custkey[idx];
        if (k >= 0 && k < dense_size) {
            const double rev = extendedprice[idx] * (1.0 - discount[idx]);
            atomicAdd(revenue_dense + k, rev);
            seen[k] = static_cast<uint8_t>(1);
        }
    }
}

__global__ void q10_seen_to_i64_kernel(
    const uint8_t* __restrict__ seen,
    int64_t* __restrict__ prefix,
    int64_t dense_size
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dense_size) {
        prefix[idx] = seen[idx] ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
    }
}

__global__ void q10_compact_sorted_kernel(
    const uint8_t* __restrict__ seen,
    const int64_t* __restrict__ prefix,
    const double* __restrict__ revenue_dense,
    int64_t* __restrict__ keys_out,
    double* __restrict__ revenue_out,
    int64_t dense_size
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < dense_size && seen[idx]) {
        const int64_t pos = prefix[idx];
        keys_out[pos] = idx;
        revenue_out[pos] = revenue_dense[idx];
    }
}

std::vector<torch::Tensor> q10_groupby_launcher(
    torch::Tensor c_custkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount
) {
    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64, "c_custkey must be int64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(c_custkey.numel() == l_extendedprice.numel(), "c_custkey and l_extendedprice length mismatch");
    TORCH_CHECK(c_custkey.numel() == l_discount.numel(), "c_custkey and l_discount length mismatch");

    if (!c_custkey.is_cuda()) {
        c_custkey = c_custkey.to(torch::kCUDA);
    }
    if (!l_extendedprice.is_cuda()) {
        l_extendedprice = l_extendedprice.to(torch::kCUDA);
    }
    if (!l_discount.is_cuda()) {
        l_discount = l_discount.to(torch::kCUDA);
    }

    c_custkey = c_custkey.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();

    const int64_t n = c_custkey.numel();
    auto key_options = c_custkey.options().dtype(torch::kInt64);
    auto val_options = l_extendedprice.options().dtype(torch::kFloat64);

    if (n == 0) {
        return {
            torch::empty({0}, key_options),
            torch::empty({0}, val_options)
        };
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t n_blocks_i64 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    TORCH_CHECK(n_blocks_i64 <= 2147483647LL, "too many CUDA blocks requested");
    const int n_blocks = static_cast<int>(n_blocks_i64);

    auto block_max = torch::empty({n_blocks_i64}, key_options);
    auto max_out = torch::empty({1}, key_options);

    q10_max_reduce_kernel<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
        c_custkey.data_ptr<int64_t>(),
        block_max.data_ptr<int64_t>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    q10_final_max_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        block_max.data_ptr<int64_t>(),
        max_out.data_ptr<int64_t>(),
        n_blocks_i64
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t max_key_h = -1;
    cudaMemcpyAsync(
        &max_key_h,
        max_out.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    );
    cudaStreamSynchronize(stream);

    if (max_key_h < 0) {
        return {
            torch::empty({0}, key_options),
            torch::empty({0}, val_options)
        };
    }

    const int64_t dense_size = max_key_h + 1;
    TORCH_CHECK(dense_size > 0, "invalid dense aggregation size");

    auto revenue_dense = torch::zeros({dense_size}, val_options);
    auto seen = torch::zeros({dense_size}, c_custkey.options().dtype(torch::kUInt8));

    q10_dense_aggregate_kernel<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
        c_custkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        revenue_dense.data_ptr<double>(),
        seen.data_ptr<uint8_t>(),
        n,
        dense_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto prefix = torch::empty({dense_size}, key_options);
    const int64_t dense_blocks_i64 = (dense_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    TORCH_CHECK(dense_blocks_i64 <= 2147483647LL, "too many dense CUDA blocks requested");
    const int dense_blocks = static_cast<int>(dense_blocks_i64);

    q10_seen_to_i64_kernel<<<dense_blocks, BLOCK_SIZE, 0, stream>>>(
        seen.data_ptr<uint8_t>(),
        prefix.data_ptr<int64_t>(),
        dense_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    thrust::device_ptr<int64_t> prefix_ptr(prefix.data_ptr<int64_t>());
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        prefix_ptr,
        prefix_ptr + dense_size,
        prefix_ptr
    );

    int64_t last_prefix_h = 0;
    uint8_t last_seen_h = 0;
    cudaMemcpyAsync(
        &last_prefix_h,
        prefix.data_ptr<int64_t>() + dense_size - 1,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    );
    cudaMemcpyAsync(
        &last_seen_h,
        seen.data_ptr<uint8_t>() + dense_size - 1,
        sizeof(uint8_t),
        cudaMemcpyDeviceToHost,
        stream
    );
    cudaStreamSynchronize(stream);

    const int64_t unique_count = last_prefix_h + (last_seen_h ? 1 : 0);

    auto keys_out = torch::empty({unique_count}, key_options);
    auto revenue_out = torch::empty({unique_count}, val_options);

    if (unique_count > 0) {
        q10_compact_sorted_kernel<<<dense_blocks, BLOCK_SIZE, 0, stream>>>(
            seen.data_ptr<uint8_t>(),
            prefix.data_ptr<int64_t>(),
            revenue_dense.data_ptr<double>(),
            keys_out.data_ptr<int64_t>(),
            revenue_out.data_ptr<double>(),
            dense_size
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {keys_out, revenue_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q10_groupby_launcher", &q10_groupby_launcher, "TPC-H Q10 dense customer revenue groupby (CUDA)");
}
'''


def _load_cuda() -> Any:
    """
    Compile and load the CUDA extension from an inlined .cu source string.

    Runs at Python import time on the host, writes device code to a temporary file,
    and asks torch's CUDA extension loader to build an sm_90 H100 binary. Keeping
    the numeric group-by in hand-written CUDA avoids the reference hot path's
    torch.unique sort plus scatter_add temporaries.
    """
    tmpdir = tempfile.mkdtemp(prefix="q10_core_cuda_")
    cu_path = os.path.join(tmpdir, "_q10_core_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q10_core_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90", "--expt-relaxed-constexpr"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT: Any = _load_cuda()


def _launch_q10_groupby(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch the CUDA dense aggregation for Q10's tensor hot path.

    GPU work: find max customer key, aggregate revenue into a dense fp64 array
    with CUDA atomics, prefix-scan the presence bitmap, then compact sorted unique
    customer keys and revenues. This replaces PyTorch eager unique+scatter_add,
    reducing intermediate tensor traffic and avoiding a general-purpose sort-heavy
    group-by for TPC-H's dense customer-key domain.
    """
    out = _EXT.q10_groupby_launcher(c_custkey, l_extendedprice, l_discount)
    return out[0], out[1]


def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU tensor core for TPC-H Q10 revenue aggregation.

    Runs entirely in the CUDA extension on GPU tensors: c_custkey is grouped and
    l_extendedprice * (1 - l_discount) is summed in float64 per customer. It is
    faster than the reference PyTorch hot path because it uses a dense customer-key
    aggregation plus one compaction pass instead of torch.unique plus scatter_add
    with larger framework overhead and temporary allocations.
    """
    return _launch_q10_groupby(c_custkey, l_extendedprice, l_discount)


def run_query(
    tables: dict[str, cudf.DataFrame],
    orderdate_lo: str = "1993-10-01",
    orderdate_hi: str = "1994-01-01",
    returnflag_val: str = "R",
    top_n: int = 20,
) -> cudf.DataFrame:
    """
    SQL: SELECT c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) AS revenue,
         c_acctbal, n_name, c_address, c_phone, c_comment
         FROM customer, orders, lineitem, nation
         WHERE c_custkey=o_custkey AND l_orderkey=o_orderkey
         AND o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01'
         AND l_returnflag='R' AND c_nationkey=n_nationkey
         GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
         ORDER BY revenue DESC LIMIT 20;
    Description: TPC-H Q10 - Returned item reporting. Finds top 20 customers by revenue
                 from returned lineitems within a quarter.
    Params:
        orderdate_lo (str): inclusive lower bound for o_orderdate.
        orderdate_hi (str): exclusive upper bound for o_orderdate.
        returnflag_val (str): lineitem return flag filter.
        top_n (int): number of top customers to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_nation = tables["nation"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter orders by date
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    # Filter lineitem by returnflag
    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with orders
    lo = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")

    # Join with customer
    loc = lo.merge(df_customer, left_on="o_custkey", right_on="c_custkey")

    # Aggregate revenue by c_custkey in torch
    custkey_t, revenue_t = _query_core(
        torch.as_tensor(loc["c_custkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(loc["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(loc["l_discount"].to_cupy(), dtype=torch.float64),
    )

    # Top-N by revenue
    k = min(int(top_n), revenue_t.shape[0])
    top_rev, top_idx = torch.topk(revenue_t, k)
    top_keys = custkey_t[top_idx]

    agg_df = cudf.DataFrame({
        "c_custkey": top_keys,
        "revenue": top_rev,
    })

    # Join back with customer to get descriptive cols
    cust_cols = df_customer[["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")

    # Join with nation
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]], left_on="c_nationkey", right_on="n_nationkey")

    # Order by revenue DESC
    agg_df = agg_df.sort_values("revenue", ascending=False).head(int(top_n))

    result = cudf.DataFrame({
        "c_custkey": agg_df["c_custkey"].reset_index(drop=True),
        "c_name": agg_df["c_name"].reset_index(drop=True),
        "revenue": agg_df["revenue"].reset_index(drop=True),
        "c_acctbal": agg_df["c_acctbal"].reset_index(drop=True),
        "n_name": agg_df["n_name"].reset_index(drop=True),
        "c_address": agg_df["c_address"].reset_index(drop=True),
        "c_phone": agg_df["c_phone"].reset_index(drop=True),
        "c_comment": agg_df["c_comment"].reset_index(drop=True),
    })
    return result
