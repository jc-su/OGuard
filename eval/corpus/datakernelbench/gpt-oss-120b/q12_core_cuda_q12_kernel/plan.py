import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused kernel that computes high/low line counts per shipmode
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// fused kernel for TPC‑H Q12
// All integer counts fit in unsigned long long (64‑bit) because they are non‑negative.
extern "C" __global__
void q12_kernel(
    const int32_t* __restrict__ shipmode_code,   // 0 = MAIL, 1 = SHIP
    const int32_t* __restrict__ priority_high,   // 1 = high priority, 0 = low
    const int32_t* __restrict__ commitdate,
    const int32_t* __restrict__ receiptdate,
    const int32_t* __restrict__ shipdate,
    const int32_t receipt_lo,
    const int32_t receipt_hi,
    unsigned long long* __restrict__ high_counts,   // length 2
    unsigned long long* __restrict__ low_counts,    // length 2
    const int64_t n_elements)
{
    // per‑block shared accumulators
    __shared__ unsigned long long block_high[2];
    __shared__ unsigned long long block_low[2];

    // initialise shared memory (only first two threads needed)
    if (threadIdx.x < 2) {
        block_high[threadIdx.x] = 0ULL;
        block_low[threadIdx.x]  = 0ULL;
    }
    __syncthreads();

    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        const int32_t cd = commitdate[idx];
        const int32_t rd = receiptdate[idx];
        const int32_t sd = shipdate[idx];
        bool keep = (cd < rd) &&
                    (sd < cd) &&
                    (rd >= receipt_lo) &&
                    (rd < receipt_hi);
        if (keep) {
            const int32_t mode = shipmode_code[idx];   // 0 or 1
            const int32_t high = priority_high[idx];   // 0 or 1
            const unsigned long long high_val = static_cast<unsigned long long>(high);
            const unsigned long long low_val  = static_cast<unsigned long long>(1 - high);
            // block‑level atomics (fast, low contention)
            atomicAdd(&block_high[mode], high_val);
            atomicAdd(&block_low[mode],  low_val);
        }
    }
    __syncthreads();

    // one thread per group adds the block's contribution to global counters
    if (threadIdx.x < 2) {
        atomicAdd(&high_counts[threadIdx.x], block_high[threadIdx.x]);
        atomicAdd(&low_counts[threadIdx.x],  block_low[threadIdx.x]);
    }
}

// Launcher returns a vector of two int64 tensors (high, low)
std::vector<torch::Tensor> q12_launcher(
    torch::Tensor shipmode_code,
    torch::Tensor priority_high,
    torch::Tensor commitdate,
    torch::Tensor receiptdate,
    torch::Tensor shipdate,
    int32_t receipt_lo,
    int32_t receipt_hi)
{
    // basic sanity checks
    TORCH_CHECK(shipmode_code.is_cuda() && shipmode_code.dtype() == torch::kInt32);
    TORCH_CHECK(priority_high.is_cuda() && priority_high.dtype() == torch::kInt32);
    TORCH_CHECK(commitdate.is_cuda() && commitdate.dtype() == torch::kInt32);
    TORCH_CHECK(receiptdate.is_cuda() && receiptdate.dtype() == torch::kInt32);
    TORCH_CHECK(shipdate.is_cuda() && shipdate.dtype() == torch::kInt32);

    // ensure contiguity
    shipmode_code = shipmode_code.contiguous();
    priority_high = priority_high.contiguous();
    commitdate = commitdate.contiguous();
    receiptdate = receiptdate.contiguous();
    shipdate = shipdate.contiguous();

    const int64_t n = shipmode_code.numel();

    // allocate output counters (int64 on GPU)
    auto opts = torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(torch::kCUDA);
    torch::Tensor high_counts = torch::zeros({2}, opts);
    torch::Tensor low_counts  = torch::zeros({2}, opts);

    const int threads = BLOCK_SIZE;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);

    // launch kernel (reinterpret cast to unsigned long long*)
    q12_kernel<<<blocks, threads>>>(
        shipmode_code.data_ptr<int32_t>(),
        priority_high.data_ptr<int32_t>(),
        commitdate.data_ptr<int32_t>(),
        receiptdate.data_ptr<int32_t>(),
        shipdate.data_ptr<int32_t>(),
        receipt_lo,
        receipt_hi,
        reinterpret_cast<unsigned long long*>(high_counts.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(low_counts.data_ptr<int64_t>()),
        n);

    // check launch success
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA launch error: ") + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return {high_counts, low_counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher,
          "TPC‑H Q12 fused high/low counts (CUDA)");
}
'''

def _load_cuda():
    """Compile the embedded CUDA source with torch.utils.cpp_extension.load."""
    tmpdir = tempfile.mkdtemp(prefix="q12_cuda_")
    cu_path = os.path.join(tmpdir, "q12_kernel.cu")
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
        verbose=False,
    )

# Load once per interpreter
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
    Invoke the fused CUDA kernel and wrap its outputs.
    Returns (group_ids, high_counts, low_counts) – all int64 tensors on the same device.
    """
    high_counts, low_counts = _EXT.q12_launcher(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
    )
    group_ids = torch.arange(2, device=l_shipmode_code.device, dtype=torch.int64)
    return group_ids, high_counts, low_counts

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
    Core computation for Q12.
    A fused CUDA kernel evaluates the filter predicates and atomically
    increments per‑ship‑mode high/low counters, eliminating multiple
    kernel launches and intermediate host‑side materialisation.
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
    SQL: SELECT l_shipmode,
            SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
            SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
         FROM orders, lineitem
         WHERE o_orderkey = l_orderkey
           AND l_shipmode IN ('MAIL','SHIP')
           AND l_commitdate < l_receiptdate
           AND l_shipdate < l_commitdate
           AND l_receiptdate >= '1994-01-01'
           AND l_receiptdate < '1995-01-01'
         GROUP BY l_shipmode
         ORDER BY l_shipmode;
    Description: TPC‑H Q12 – counts urgent vs non‑urgent orders by ship mode.
    Params:
      shipmode1, shipmode2 (str): allowed shipping modes.
      receipt_lo, receipt_hi (str): receipt date range.
      priority1, priority2 (str): high‑priority order priority values.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    # Pre‑filter lineitem on shipmode to shrink the join
    li = df_lineitem[df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])]

    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey")

    # Encode shipmode: MAIL=0, SHIP=1
    shipmode_code = (joined["l_shipmode"] == shipmode2).astype("int32")
    priority_high = (
        (joined["o_orderpriority"] == priority1)
        | (joined["o_orderpriority"] == priority2)
    ).astype("int32")

    # Convert dates to int32 epoch days and move to GPU
    l_commit_t = torch.as_tensor(
        ((joined["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        device="cuda",
    )
    l_receipt_t = torch.as_tensor(
        ((joined["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        device="cuda",
    )
    l_ship_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        device="cuda",
    )
    shipmode_t = torch.as_tensor(
        shipmode_code.to_cupy(), dtype=torch.int32, device="cuda"
    )
    priority_t = torch.as_tensor(
        priority_high.to_cupy(), dtype=torch.int32, device="cuda"
    )

    group_ids, high_counts, low_counts = _query_core(
        shipmode_t,
        priority_t,
        l_commit_t,
        l_receipt_t,
        l_ship_t,
        receipt_lo_days,
        receipt_hi_days,
    )

    # Produce ordered result (MAIL < SHIP alphabetically)
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
