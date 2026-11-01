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
    const int32_t* __restrict__ l_commitdate,
    const int32_t* __restrict__ l_receiptdate,
    const int32_t* __restrict__ l_shipdate,
    const int32_t* __restrict__ l_shipmode,
    const int32_t* __restrict__ o_orderpriority_high,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days,
    int* __restrict__ out,
    int64_t n_elements
) {
    __shared__ int block_counts[4]; // 4 counters per block

    // Initialize shared memory to 0 by thread 0
    if (threadIdx.x == 0) {
        block_counts[0] = 0;
        block_counts[1] = 0;
        block_counts[2] = 0;
        block_counts[3] = 0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t commit = l_commitdate[idx];
        int32_t receipt = l_receiptdate[idx];
        int32_t ship = l_shipdate[idx];
        int32_t mode = l_shipmode[idx];
        int32_t high = o_orderpriority_high[idx];

        bool keep = (commit < receipt) && (ship < commit) && (receipt >= receipt_lo_days) && (receipt < receipt_hi_days);

        if (keep) {
            // mode: 0 for MAIL, 1 for SHIP (encoded as in reference: shipmode1=MAIL=0, shipmode2=SHIP=1)
            // high: 1 for high priority, 0 for low
            // high counts come before low counts in output (matching reference: high then low)
            int idx_count = mode * 2 + (1 - high);
            atomicAdd(&block_counts[idx_count], 1);
        }
    }

    __syncthreads();

    // Add block counts to global output
    if (threadIdx.x == 0) {
        atomicAdd(&out[0], block_counts[0]);
        atomicAdd(&out[1], block_counts[1]);
        atomicAdd(&out[2], block_counts[2]);
        atomicAdd(&out[3], block_counts[3]);
    }
}

torch::Tensor q12_launcher(
    torch::Tensor l_commitdate,
    torch::Tensor l_receiptdate,
    torch::Tensor l_shipdate,
    torch::Tensor l_shipmode,
    torch::Tensor o_orderpriority_high,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days
) {
    TORCH_CHECK(l_commitdate.scalar_type() == torch::kInt32);
    TORCH_CHECK(l_receiptdate.scalar_type() == torch::kInt32);
    TORCH_CHECK(l_shipdate.scalar_type() == torch::kInt32);
    TORCH_CHECK(l_shipmode.scalar_type() == torch::kInt32);
    TORCH_CHECK(o_orderpriority_high.scalar_type() == torch::kInt32);

    if (!l_commitdate.is_cuda()) l_commitdate = l_commitdate.cuda();
    if (!l_receiptdate.is_cuda()) l_receiptdate = l_receiptdate.cuda();
    if (!l_shipdate.is_cuda()) l_shipdate = l_shipdate.cuda();
    if (!l_shipmode.is_cuda()) l_shipmode = l_shipmode.cuda();
    if (!o_orderpriority_high.is_cuda()) o_orderpriority_high = o_orderpriority_high.cuda();

    l_commitdate = l_commitdate.contiguous();
    l_receiptdate = l_receiptdate.contiguous();
    l_shipdate = l_shipdate.contiguous();
    l_shipmode = l_shipmode.contiguous();
    o_orderpriority_high = o_orderpriority_high.contiguous();

    int64_t n = l_commitdate.numel();
    // Use int32 for output since counts fit in 32 bits (max ~60M rows)
    auto out = torch::zeros({4}, l_commitdate.options().dtype(torch::kInt32));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q12_kernel<<<blocks, BLOCK_SIZE>>>(
        l_commitdate.data_ptr<int32_t>(),
        l_receiptdate.data_ptr<int32_t>(),
        l_shipdate.data_ptr<int32_t>(),
        l_shipmode.data_ptr<int32_t>(),
        o_orderpriority_high.data_ptr<int32_t>(),
        receipt_lo_days,
        receipt_hi_days,
        out.data_ptr<int>(),
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher, "TPC-H Q12 launcher (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q12_cuda_")
    cu_path = os.path.join(tmpdir, "_q12_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q12_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

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
    Description: TPC-H Q12 - counts urgent vs non-urgent orders by ship mode. Uses cuDF for join and filtering, then CUDA for fused filter+aggregate.
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

    # Call CUDA kernel
    result_tensor = _EXT.q12_launcher(
        l_commit_t,
        l_receipt_t,
        l_ship_t,
        shipmode_t,
        priority_t,
        receipt_lo_days,
        receipt_hi_days,
    )

    # result_tensor is [high_mail, low_mail, high_ship, low_ship]
    # We need to return in order: MAIL, then SHIP
    result_host = result_tensor.cpu().tolist()
    high_mail = result_host[0]
    low_mail = result_host[1]
    high_ship = result_host[2]
    low_ship = result_host[3]

    result = cudf.DataFrame({
        "l_shipmode": [shipmode1, shipmode2],
        "high_line_count": [high_mail, high_ship],
        "low_line_count": [low_mail, low_ship],
    })
    return result
