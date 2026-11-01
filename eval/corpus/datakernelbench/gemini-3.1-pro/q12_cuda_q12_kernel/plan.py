import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q12_kernel(
    const int32_t* __restrict__ shipmode_ptr,
    const int32_t* __restrict__ priority_ptr,
    const int32_t* __restrict__ commitdate_ptr,
    const int32_t* __restrict__ receiptdate_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t receipt_lo,
    int32_t receipt_hi,
    int64_t* __restrict__ out_high,
    int64_t* __restrict__ out_low,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Thread-local accumulation for 2 shipmodes (0=MAIL, 1=SHIP)
    int64_t h[2] = {0, 0};
    int64_t l[2] = {0, 0};
    
    // Grid-stride loop
    while (idx < n_elements) {
        int32_t commit = commitdate_ptr[idx];
        int32_t receipt = receiptdate_ptr[idx];
        int32_t ship = shipdate_ptr[idx];
        
        if (commit < receipt && ship < commit && receipt >= receipt_lo && receipt < receipt_hi) {
            int32_t sm = shipmode_ptr[idx];
            int32_t pr = priority_ptr[idx];
            
            if (pr == 1) {
                h[sm]++;
            } else {
                l[sm]++;
            }
        }
        idx += stride;
    }

    // Block-level reduction in shared memory
    __shared__ int64_t sh_h0[BLOCK_SIZE];
    __shared__ int64_t sh_h1[BLOCK_SIZE];
    __shared__ int64_t sh_l0[BLOCK_SIZE];
    __shared__ int64_t sh_l1[BLOCK_SIZE];

    sh_h0[threadIdx.x] = h[0];
    sh_h1[threadIdx.x] = h[1];
    sh_l0[threadIdx.x] = l[0];
    sh_l1[threadIdx.x] = l[1];
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sh_h0[threadIdx.x] += sh_h0[threadIdx.x + offset];
            sh_h1[threadIdx.x] += sh_h1[threadIdx.x + offset];
            sh_l0[threadIdx.x] += sh_l0[threadIdx.x + offset];
            sh_l1[threadIdx.x] += sh_l1[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Global atomics from thread 0
    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)&out_high[0], (unsigned long long)sh_h0[0]);
        atomicAdd((unsigned long long*)&out_high[1], (unsigned long long)sh_h1[0]);
        atomicAdd((unsigned long long*)&out_low[0], (unsigned long long)sh_l0[0]);
        atomicAdd((unsigned long long*)&out_low[1], (unsigned long long)sh_l1[0]);
    }
}

std::vector<torch::Tensor> q12_launcher(
    torch::Tensor shipmode,
    torch::Tensor priority,
    torch::Tensor commitdate,
    torch::Tensor receiptdate,
    torch::Tensor shipdate,
    int32_t receipt_lo,
    int32_t receipt_hi
) {
    int64_t n = shipmode.numel();
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(shipmode.device());
    auto out_high = torch::zeros({2}, options);
    auto out_low = torch::zeros({2}, options);

    if (n > 0) {
        int blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 8192);
        q12_kernel<<<blocks, BLOCK_SIZE>>>(
            shipmode.data_ptr<int32_t>(),
            priority.data_ptr<int32_t>(),
            commitdate.data_ptr<int32_t>(),
            receiptdate.data_ptr<int32_t>(),
            shipdate.data_ptr<int32_t>(),
            receipt_lo,
            receipt_hi,
            out_high.data_ptr<int64_t>(),
            out_low.data_ptr<int64_t>(),
            n
        );
    }
    
    return {out_high, out_low};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher, "TPC-H Q12 kernel launcher");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q12_cuda_")
    cu_path = os.path.join(tmpdir, "q12_kernels.cu")
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
    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey", how="inner")

    # Encode shipmode: shipmode1=0, shipmode2=1
    shipmode_code = (joined["l_shipmode"] == shipmode2).astype("int32")
    priority_high = (
        (joined["o_orderpriority"] == priority1) | 
        (joined["o_orderpriority"] == priority2)
    ).astype("int32")

    # Extract days as int32
    l_commit_t = torch.as_tensor(
        ((joined["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    ).contiguous()
    l_receipt_t = torch.as_tensor(
        ((joined["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    ).contiguous()
    l_ship_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    ).contiguous()
    
    shipmode_t = torch.as_tensor(shipmode_code.to_cupy(), dtype=torch.int32).contiguous()
    priority_t = torch.as_tensor(priority_high.to_cupy(), dtype=torch.int32).contiguous()

    out_high, out_low = _EXT.q12_launcher(
        shipmode_t,
        priority_t,
        l_commit_t,
        l_receipt_t,
        l_ship_t,
        receipt_lo_days,
        receipt_hi_days
    )

    names = [shipmode1, shipmode2]
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = out_high.cpu().tolist()
    lc_host = out_low.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count": [lc_host[i] for i in ordered],
    })
    
    return result
