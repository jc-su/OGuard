import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void q12_kernel(
    const int32_t* __restrict__ shipmode_ptr,
    const int32_t* __restrict__ priority_ptr,
    const int32_t* __restrict__ commit_ptr,
    const int32_t* __restrict__ receipt_ptr,
    const int32_t* __restrict__ ship_ptr,
    int32_t receipt_lo,
    int32_t receipt_hi,
    unsigned long long* __restrict__ counts_ptr,
    int64_t n
) {
    extern __shared__ unsigned long long s_counts[];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) s_counts[i] = 0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long l_counts[4] = {0, 0, 0, 0};

    if (idx < n) {
        int32_t commit = commit_ptr[idx];
        int32_t receipt = receipt_ptr[idx];
        int32_t ship = ship_ptr[idx];
        
        bool date_ok = (commit < receipt) && (ship < commit) && 
                       (receipt >= receipt_lo) && (receipt < receipt_hi);
        
        if (date_ok) {
            int32_t mode = shipmode_ptr[idx];
            int32_t prio = priority_ptr[idx];
            int bin = mode * 2 + prio;
            l_counts[bin]++;
        }
    }

    for (int i = 0; i < 4; ++i) {
        if (l_counts[i] > 0) {
            atomicAdd(&s_counts[i], l_counts[i]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            if (s_counts[i] > 0) {
                atomicAdd(&counts_ptr[i], s_counts[i]);
            }
        }
    }
}

void q12_launcher(
    torch::Tensor shipmode,
    torch::Tensor priority,
    torch::Tensor commit,
    torch::Tensor receipt,
    torch::Tensor ship,
    int32_t receipt_lo,
    int32_t receipt_hi,
    torch::Tensor counts
) {
    TORCH_CHECK(shipmode.is_cuda(), "shipmode must be CUDA");
    TORCH_CHECK(shipmode.scalar_type() == torch::kInt32, "shipmode must be int32");
    TORCH_CHECK(priority.scalar_type() == torch::kInt32, "priority must be int32");
    TORCH_CHECK(commit.scalar_type() == torch::kInt32, "commit must be int32");
    TORCH_CHECK(receipt.scalar_type() == torch::kInt32, "receipt must be int32");
    TORCH_CHECK(ship.scalar_type() == torch::kInt32, "ship must be int32");
    TORCH_CHECK(counts.scalar_type() == torch::kInt64, "counts must be int64");

    shipmode = shipmode.contiguous();
    priority = priority.contiguous();
    commit = commit.contiguous();
    receipt = receipt.contiguous();
    ship = ship.contiguous();
    counts = counts.contiguous();

    int64_t n = shipmode.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem = 4 * sizeof(unsigned long long);

    q12_kernel<<<blocks, BLOCK_SIZE, smem>>>(
        shipmode.data_ptr<int32_t>(),
        priority.data_ptr<int32_t>(),
        commit.data_ptr<int32_t>(),
        receipt.data_ptr<int32_t>(),
        ship.data_ptr<int32_t>(),
        receipt_lo, receipt_hi,
        reinterpret_cast<unsigned long long*>(counts.data_ptr<int64_t>()),
        n
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher, "TPC-H Q12 counts (CUDA)");
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
    Fused CUDA kernel for TPC-H Q12.
    Runs filter (date predicates) + project (shipmode/priority encoding) + aggregate (counts per mode/priority)
    in a single GPU kernel pass, reducing HBM traffic vs multiple cuDF/PyTorch operations.
    
    Bin layout: bin = mode * 2 + prio
    - counts[0]: MAIL (mode=0), low priority (prio=0)
    - counts[1]: MAIL (mode=0), high priority (prio=1)
    - counts[2]: SHIP (mode=1), low priority (prio=0)
    - counts[3]: SHIP (mode=1), high priority (prio=1)
    """
    n = l_shipmode_code.numel()
    counts = torch.zeros(4, dtype=torch.int64, device=l_shipmode_code.device)
    
    _EXT.q12_launcher(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
        counts,
    )
    
    group_ids = torch.arange(2, device=counts.device, dtype=torch.int64)
    # high priority is prio=1, so counts[1] and counts[3]
    high_counts = torch.stack([counts[1], counts[3]])
    # low priority is prio=0, so counts[0] and counts[2]
    low_counts = torch.stack([counts[0], counts[2]])
    
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
    SQL: SELECT l_shipmode, SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
         SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
         FROM orders, lineitem WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL','SHIP')
         AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
         AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01'
         GROUP BY l_shipmode ORDER BY l_shipmode;
    Description: TPC-H Q12 - counts urgent vs non-urgent orders by ship mode.
                 cuDF handles join and string encoding; fused CUDA kernel performs
                 date filtering and grouped counting in one pass for memory-bound speedup.
    Params:
      shipmode1, shipmode2 (str): allowed shipping modes.
      receipt_lo, receipt_hi (str): receipt date range.
      priority1, priority2 (str): high-priority order priority values.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    li = df_lineitem[df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])]
    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey")

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

    names = [shipmode1, shipmode2]
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_counts.cpu().tolist()
    lc_host = low_counts.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count": [lc_host[i] for i in ordered],
    })
    return result
