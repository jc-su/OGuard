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
    const int32_t* __restrict__ commit_ptr,
    const int32_t* __restrict__ receipt_ptr,
    const int32_t* __restrict__ ship_ptr,
    int32_t receipt_lo,
    int32_t receipt_hi,
    unsigned long long* __restrict__ out_counts,
    int64_t n_elements
) {
    __shared__ unsigned long long s_counts[4];
    
    // Initialize shared memory
    if (threadIdx.x < 4) {
        s_counts[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Each thread accumulates locally
    unsigned long long my_counts[4] = {0, 0, 0, 0};
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t commit = commit_ptr[idx];
        int32_t receipt = receipt_ptr[idx];
        int32_t ship = ship_ptr[idx];
        
        bool keep = (commit < receipt) && (ship < commit) &&
                    (receipt >= receipt_lo) && (receipt < receipt_hi);
        
        if (keep) {
            int32_t mode = shipmode_ptr[idx];
            int32_t high = priority_ptr[idx];
            
            if (mode >= 0 && mode < 2) {
                // bucket layout: [mode0_high, mode0_low, mode1_high, mode1_low]
                int32_t bucket = mode * 2 + (high ? 0 : 1);
                my_counts[bucket]++;
            }
        }
    }
    
    // Add local counts to shared memory using atomics
    for (int b = 0; b < 4; b++) {
        atomicAdd(&s_counts[b], my_counts[b]);
    }
    __syncthreads();
    
    // Thread 0 writes to global output
    if (threadIdx.x == 0) {
        atomicAdd(&out_counts[0], s_counts[0]);
        atomicAdd(&out_counts[1], s_counts[1]);
        atomicAdd(&out_counts[2], s_counts[2]);
        atomicAdd(&out_counts[3], s_counts[3]);
    }
}

torch::Tensor q12_launcher(
    torch::Tensor shipmode,
    torch::Tensor priority,
    torch::Tensor commit,
    torch::Tensor receipt,
    torch::Tensor ship,
    int32_t receipt_lo,
    int32_t receipt_hi
) {
    TORCH_CHECK(shipmode.scalar_type() == torch::kInt32);
    TORCH_CHECK(priority.scalar_type() == torch::kInt32);
    TORCH_CHECK(commit.scalar_type() == torch::kInt32);
    TORCH_CHECK(receipt.scalar_type() == torch::kInt32);
    TORCH_CHECK(ship.scalar_type() == torch::kInt32);
    
    if (!shipmode.is_cuda()) shipmode = shipmode.cuda();
    if (!priority.is_cuda()) priority = priority.cuda();
    if (!commit.is_cuda()) commit = commit.cuda();
    if (!receipt.is_cuda()) receipt = receipt.cuda();
    if (!ship.is_cuda()) ship = ship.cuda();
    
    shipmode = shipmode.contiguous();
    priority = priority.contiguous();
    commit = commit.contiguous();
    receipt = receipt.contiguous();
    ship = ship.contiguous();
    
    int64_t n = shipmode.numel();
    auto out = torch::zeros({4}, shipmode.options().dtype(torch::kInt64));
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q12_kernel<<<blocks, BLOCK_SIZE>>>(
        shipmode.data_ptr<int32_t>(),
        priority.data_ptr<int32_t>(),
        commit.data_ptr<int32_t>(),
        receipt.data_ptr<int32_t>(),
        ship.data_ptr<int32_t>(),
        receipt_lo, receipt_hi,
        reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()),
        n
    );
    
    return out;
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

def _launch_q12(
    l_shipmode_code: torch.Tensor,
    o_orderpriority_high: torch.Tensor,
    l_commitdate: torch.Tensor,
    l_receiptdate: torch.Tensor,
    l_shipdate: torch.Tensor,
    receipt_lo_days: int,
    receipt_hi_days: int,
) -> torch.Tensor:
    """
    Launch CUDA kernel for Q12 aggregation.
    
    Fused kernel: filter (date predicates) + project (shipmode/priority) + aggregate (4 counters)
    in a single GPU pass. Uses shared-memory accumulation with atomics.
    """
    return _EXT.q12_launcher(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
    )

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
    GPU compute core for TPC-H Q12 using fused CUDA kernel.
    
    Replaces reference eager torch operations with single CUDA kernel that:
    1. Filters rows by date predicates in registers (no intermediate mask tensor)
    2. Projects shipmode (0/1) and priority (0/1) to 4 buckets
    3. Aggregates via shared-memory accumulation + atomicAdd
    
    Memory-bound optimization: one HBM pass instead of multiple cuDF/torch passes.
    
    Returns:
        group_ids: torch.Tensor([0, 1]) - shipmode group identifiers
        high_counts: torch.Tensor([MAIL_high, SHIP_high])
        low_counts: torch.Tensor([MAIL_low, SHIP_low])
    """
    counts = _launch_q12(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
    )
    
    # CUDA kernel outputs: [mode0_high, mode0_low, mode1_high, mode1_low]
    # Reference expects: high_counts=[mode0_high, mode1_high], low_counts=[mode0_low, mode1_low]
    group_ids = torch.arange(2, device=counts.device, dtype=torch.int64)
    high_counts = torch.stack([counts[0], counts[2]])
    low_counts = torch.stack([counts[1], counts[3]])
    
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
