import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q12_kernel(
    const int32_t* __restrict__ shipmode_code_ptr,
    const int32_t* __restrict__ priority_high_ptr,
    const int32_t* __restrict__ l_commitdate_ptr,
    const int32_t* __restrict__ l_receiptdate_ptr,
    const int32_t* __restrict__ l_shipdate_ptr,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days,
    int64_t* __restrict__ high0_out_ptr,
    int64_t* __restrict__ high1_out_ptr,
    int64_t* __restrict__ low0_out_ptr,
    int64_t* __restrict__ low1_out_ptr,
    int64_t n_elements
) {
    __shared__ int64_t block_high0[BLOCK_SIZE];
    __shared__ int64_t block_high1[BLOCK_SIZE];
    __shared__ int64_t block_low0[BLOCK_SIZE];
    __shared__ int64_t block_low1[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t my_high0 = 0;
    int64_t my_high1 = 0;
    int64_t my_low0 = 0;
    int64_t my_low1 = 0;

    if (idx < n_elements) {
        int32_t commit = l_commitdate_ptr[idx];
        int32_t receipt = l_receiptdate_ptr[idx];
        int32_t ship = l_shipdate_ptr[idx];
        int32_t code = shipmode_code_ptr[idx];
        int32_t prio = priority_high_ptr[idx];

        if (commit < receipt && ship < commit && receipt >= receipt_lo_days && receipt < receipt_hi_days) {
            if (code == 0) {
                if (prio) my_high0 = 1;
                else my_low0 = 1;
            } else {
                if (prio) my_high1 = 1;
                else my_low1 = 1;
            }
        }
    }

    block_high0[threadIdx.x] = my_high0;
    block_high1[threadIdx.x] = my_high1;
    block_low0[threadIdx.x] = my_low0;
    block_low1[threadIdx.x] = my_low1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_high0[threadIdx.x] += block_high0[threadIdx.x + s];
            block_high1[threadIdx.x] += block_high1[threadIdx.x + s];
            block_low0[threadIdx.x] += block_low0[threadIdx.x + s];
            block_low1[threadIdx.x] += block_low1[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)high0_out_ptr, (unsigned long long)block_high0[0]);
        atomicAdd((unsigned long long*)high1_out_ptr, (unsigned long long)block_high1[0]);
        atomicAdd((unsigned long long*)low0_out_ptr, (unsigned long long)block_low0[0]);
        atomicAdd((unsigned long long*)low1_out_ptr, (unsigned long long)block_low1[0]);
    }
}

std::vector<torch::Tensor> q12_launcher(
    torch::Tensor shipmode_code,
    torch::Tensor priority_high,
    torch::Tensor l_commitdate,
    torch::Tensor l_receiptdate,
    torch::Tensor l_shipdate,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days
) {
    TORCH_CHECK(shipmode_code.scalar_type() == torch::kInt32);
    TORCH_CHECK(priority_high.scalar_type() == torch::kInt32);
    TORCH_CHECK(l_commitdate.scalar_type() == torch::kInt32);
    TORCH_CHECK(l_receiptdate.scalar_type() == torch::kInt32);
    TORCH_CHECK(l_shipdate.scalar_type() == torch::kInt32);

    if (!shipmode_code.is_cuda()) shipmode_code = shipmode_code.cuda();
    if (!priority_high.is_cuda()) priority_high = priority_high.cuda();
    if (!l_commitdate.is_cuda()) l_commitdate = l_commitdate.cuda();
    if (!l_receiptdate.is_cuda()) l_receiptdate = l_receiptdate.cuda();
    if (!l_shipdate.is_cuda()) l_shipdate = l_shipdate.cuda();

    shipmode_code = shipmode_code.contiguous();
    priority_high = priority_high.contiguous();
    l_commitdate = l_commitdate.contiguous();
    l_receiptdate = l_receiptdate.contiguous();
    l_shipdate = l_shipdate.contiguous();

    int64_t n = shipmode_code.numel();
    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
    auto high0 = torch::zeros({1}, opts);
    auto high1 = torch::zeros({1}, opts);
    auto low0 = torch::zeros({1}, opts);
    auto low1 = torch::zeros({1}, opts);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q12_kernel<<<blocks, BLOCK_SIZE>>>(
        shipmode_code.data_ptr<int32_t>(),
        priority_high.data_ptr<int32_t>(),
        l_commitdate.data_ptr<int32_t>(),
        l_receiptdate.data_ptr<int32_t>(),
        l_shipdate.data_ptr<int32_t>(),
        receipt_lo_days, receipt_hi_days,
        high0.data_ptr<int64_t>(),
        high1.data_ptr<int64_t>(),
        low0.data_ptr<int64_t>(),
        low1.data_ptr<int64_t>(),
        n
    );

    return {high0, high1, low0, low1};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher, "TPC-H Q12 CUDA kernel: filter + group-by shipmode (2 groups)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q12_cuda_")
    cu_path = os.path.join(tmpdir, "_q12_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    ext = load(
        name="q12_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )
    return ext

_EXT = _load_cuda()

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
    GPU path: cuDF handles the join and string filtering, then a fused CUDA kernel
    applies all filters and aggregates into two group bins (mail=0, ship=1) in one pass.
    This avoids intermediate materialization and multiple kernel launches.
    Params:
      shipmode1, shipmode2 (str): allowed shipping modes.
      receipt_lo, receipt_hi (str): receipt date range.
      priority1, priority2 (str): high-priority order priority values.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    # Pre-filter lineitem on shipmode to reduce join size
    li = df_lineitem[df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])]

    # Merge on orderkey
    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey")

    # Encode shipmode: MAIL=0, SHIP=1
    shipmode_code = (joined["l_shipmode"] == shipmode2).astype("int32")
    priority_high = (
        (joined["o_orderpriority"] == priority1)
        | (joined["o_orderpriority"] == priority2)
    ).astype("int32")

    # Convert dates to int32 epoch days for the kernel
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

    # Launch fused CUDA kernel: filter + group-by
    # Returns [high0, high1, low0, low1]
    result_tensors = _EXT.q12_launcher(
        shipmode_t, priority_t, l_commit_t, l_receipt_t, l_ship_t,
        receipt_lo_days, receipt_hi_days,
    )

    high0 = result_tensors[0].item()
    high1 = result_tensors[1].item()
    low0 = result_tensors[2].item()
    low1 = result_tensors[3].item()

    # Decode group ids back to strings; order alphabetically (MAIL < SHIP)
    names = [shipmode1, shipmode2]
    ordered = sorted(range(2), key=lambda i: names[i])
    
    # Build counts in alphabetical order
    counts_high = {0: high0, 1: high1}
    counts_low = {0: low0, 1: low1}
    
    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [int(counts_high[i]) for i in ordered],
        "low_line_count": [int(counts_low[i]) for i in ordered],
    })
    return result
