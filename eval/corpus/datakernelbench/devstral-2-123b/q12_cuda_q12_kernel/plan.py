import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_bf16.hpp>

#define BLOCK_SIZE 1024
#define GROUP_COUNT 2

__global__ void q12_kernel(
    const int32_t* __restrict__ shipmode_code,
    const int32_t* __restrict__ priority_high,
    const int32_t* __restrict__ commit_days,
    const int32_t* __restrict__ receipt_days,
    const int32_t* __restrict__ ship_days,
    int32_t date_lo,
    int32_t date_hi,
    unsigned long long* __restrict__ high_out,
    unsigned long long* __restrict__ low_out,
    int64_t n_elements
) {
    __shared__ unsigned long long block_high_counts[GROUP_COUNT];
    __shared__ unsigned long long block_low_counts[GROUP_COUNT];

    for (int i = 0; i < GROUP_COUNT; i++) {
        if (threadIdx.x < GROUP_COUNT) {
            block_high_counts[threadIdx.x] = 0;
            block_low_counts[threadIdx.x] = 0;
        }
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements)
        return;

    int32_t c_date = commit_days[idx];
    int32_t r_date = receipt_days[idx];
    int32_t s_date = ship_days[idx];
    int32_t sm = shipmode_code[idx];
    int32_t h_flag = priority_high[idx];

    bool keep = (c_date < r_date)
              && (s_date < c_date)
              && (r_date >= date_lo)
              && (r_date < date_hi);

    if (keep) {
        unsigned long long thread_high = h_flag;
        unsigned long long thread_low = 1 - h_flag;

        atomicAdd(&block_high_counts[sm], thread_high);
        atomicAdd(&block_low_counts[sm], thread_low);
    }
    __syncthreads();

    if (threadIdx.x < GROUP_COUNT) {
        atomicAdd(&high_out[threadIdx.x], block_high_counts[threadIdx.x]);
        atomicAdd(&low_out[threadIdx.x], block_low_counts[threadIdx.x]);
    }
}

void q12_launcher(
    torch::Tensor shipmode_code,
    torch::Tensor priority_high,
    torch::Tensor commit_days,
    torch::Tensor receipt_days,
    torch::Tensor ship_days,
    int32_t date_lo,
    int32_t date_hi,
    torch::Tensor high_out,
    torch::Tensor low_out
) {
    TORCH_CHECK(shipmode_code.scalar_type() == torch::kInt32);
    TORCH_CHECK(priority_high.scalar_type() == torch::kInt32);
    TORCH_CHECK(commit_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(receipt_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(ship_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(high_out.scalar_type() == torch::kLong);
    TORCH_CHECK(low_out.scalar_type() == torch::kLong);

    if (!shipmode_code.is_cuda()) shipmode_code = shipmode_code.cuda();
    if (!priority_high.is_cuda()) priority_high = priority_high.cuda();
    if (!commit_days.is_cuda()) commit_days = commit_days.cuda();
    if (!receipt_days.is_cuda()) receipt_days = receipt_days.cuda();
    if (!ship_days.is_cuda()) ship_days = ship_days.cuda();
    if (!high_out.is_cuda()) high_out = high_out.cuda();
    if (!low_out.is_cuda()) low_out = low_out.cuda();

    shipmode_code = shipmode_code.contiguous();
    priority_high = priority_high.contiguous();
    commit_days = commit_days.contiguous();
    receipt_days = receipt_days.contiguous();
    ship_days = ship_days.contiguous();
    high_out = high_out.contiguous();
    low_out = low_out.contiguous();

    int64_t n = shipmode_code.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q12_kernel<<<blocks, BLOCK_SIZE>>>(
        shipmode_code.data_ptr<int32_t>(),
        priority_high.data_ptr<int32_t>(),
        commit_days.data_ptr<int32_t>(),
        receipt_days.data_ptr<int32_t>(),
        ship_days.data_ptr<int32_t>(),
        date_lo, date_hi,
        (unsigned long long*)high_out.data_ptr<int64_t>(),
        (unsigned long long*)low_out.data_ptr<int64_t>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher, "TPC-H Q12 fused scan (CUDA)");
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
        name="q12_cuda_ext_v2",
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

    high_out = torch.zeros(2, dtype=torch.int64, device=shipmode_t.device)
    low_out = torch.zeros(2, dtype=torch.int64, device=shipmode_t.device)

    _EXT.q12_launcher(
        shipmode_t, priority_t, l_commit_t, l_receipt_t, l_ship_t,
        receipt_lo_days, receipt_hi_days, high_out, low_out
    )

    # Decode group ids back to strings; order alphabetically (MAIL < SHIP)
    names = [shipmode1, shipmode2]  # index 0 -> MAIL, 1 -> SHIP
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_out.cpu().tolist()
    lc_host = low_out.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count": [lc_host[i] for i in ordered],
    })
    return result
