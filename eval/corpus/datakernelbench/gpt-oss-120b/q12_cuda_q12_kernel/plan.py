import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: one fused kernel that filters, groups, and counts
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ----------------------------------------------------------------------
// One‑pass kernel for TPC‑H Q12
// ----------------------------------------------------------------------
extern "C" __global__
void q12_kernel(
    const int32_t* __restrict__ shipmode,   // 0=shipmode1, 1=shipmode2
    const int32_t* __restrict__ priority,   // 1 if high‑priority else 0
    const int32_t* __restrict__ commit,
    const int32_t* __restrict__ receipt,
    const int32_t* __restrict__ ship,
    int32_t receipt_lo,
    int32_t receipt_hi,
    unsigned long long* __restrict__ high_out,   // length 2
    unsigned long long* __restrict__ low_out,    // length 2
    const int64_t N)
{
    // per‑thread tallies (keep as 0/1)
    unsigned long long h0 = 0ULL, h1 = 0ULL;
    unsigned long long l0 = 0ULL, l1 = 0ULL;

    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < N) {
        int32_t c = commit[idx];
        int32_t r = receipt[idx];
        int32_t s = ship[idx];
        bool keep = (c < r) && (s < c) && (r >= receipt_lo) && (r < receipt_hi);
        if (keep) {
            int32_t code = shipmode[idx];      // 0 or 1
            int32_t high = priority[idx];      // 1 for high, 0 otherwise
            int32_t low  = 1 - high;
            if (code == 0) {
                h0 = (unsigned long long)high;
                l0 = (unsigned long long)low;
            } else {
                h1 = (unsigned long long)high;
                l1 = (unsigned long long)low;
            }
        }
    }

    // Shared memory buffers for reduction (one per group)
    __shared__ unsigned long long sh_h0[BLOCK_SIZE];
    __shared__ unsigned long long sh_h1[BLOCK_SIZE];
    __shared__ unsigned long long sh_l0[BLOCK_SIZE];
    __shared__ unsigned long long sh_l1[BLOCK_SIZE];

    sh_h0[threadIdx.x] = h0;
    sh_h1[threadIdx.x] = h1;
    sh_l0[threadIdx.x] = l0;
    sh_l1[threadIdx.x] = l1;
    __syncthreads();

    // Tree reduction inside the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_h0[threadIdx.x] += sh_h0[threadIdx.x + stride];
            sh_h1[threadIdx.x] += sh_h1[threadIdx.x + stride];
            sh_l0[threadIdx.x] += sh_l0[threadIdx.x + stride];
            sh_l1[threadIdx.x] += sh_l1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // One atomic add per group per block
    if (threadIdx.x == 0) {
        atomicAdd(high_out + 0, sh_h0[0]);
        atomicAdd(high_out + 1, sh_h1[0]);
        atomicAdd(low_out  + 0, sh_l0[0]);
        atomicAdd(low_out  + 1, sh_l1[0]);
    }
}

// ----------------------------------------------------------------------
// Launcher exposed to Python
// ----------------------------------------------------------------------
void q12_launcher(
    torch::Tensor shipmode,
    torch::Tensor priority,
    torch::Tensor commit,
    torch::Tensor receipt,
    torch::Tensor ship,
    int32_t receipt_lo,
    int32_t receipt_hi,
    torch::Tensor high_out,
    torch::Tensor low_out)
{
    // ----- input validation -----
    TORCH_CHECK(shipmode.is_cuda() && priority.is_cuda() && commit.is_cuda()
                && receipt.is_cuda() && ship.is_cuda(),
                "All input tensors must reside on CUDA");
    TORCH_CHECK(high_out.is_cuda() && low_out.is_cuda(),
                "Output tensors must reside on CUDA");
    TORCH_CHECK(shipmode.dtype()   == torch::kInt32, "shipmode dtype must be int32");
    TORCH_CHECK(priority.dtype()   == torch::kInt32, "priority dtype must be int32");
    TORCH_CHECK(commit.dtype()     == torch::kInt32, "commit dtype must be int32");
    TORCH_CHECK(receipt.dtype()    == torch::kInt32, "receipt dtype must be int32");
    TORCH_CHECK(ship.dtype()       == torch::kInt32, "ship dtype must be int32");
    TORCH_CHECK(high_out.dtype()   == torch::kInt64, "high_out dtype must be int64");
    TORCH_CHECK(low_out.dtype()    == torch::kInt64, "low_out dtype must be int64");
    TORCH_CHECK(high_out.numel() == 2 && low_out.numel() == 2,
                "high_out / low_out must have length 2");

    // ----- ensure contiguity -----
    shipmode = shipmode.contiguous();
    priority = priority.contiguous();
    commit   = commit.contiguous();
    receipt  = receipt.contiguous();
    ship     = ship.contiguous();

    const int64_t N = shipmode.numel();
    const int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Zero outputs (caller may have done this, but for safety)
    high_out.zero_();
    low_out.zero_();

    // Launch kernel
    q12_kernel<<<blocks, BLOCK_SIZE>>>(
        shipmode.data_ptr<int32_t>(),
        priority.data_ptr<int32_t>(),
        commit.data_ptr<int32_t>(),
        receipt.data_ptr<int32_t>(),
        ship.data_ptr<int32_t>(),
        receipt_lo,
        receipt_hi,
        reinterpret_cast<unsigned long long*>(high_out.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(low_out.data_ptr<int64_t>()),
        N);

    // Check launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in q12_kernel: ", cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q12_launcher", &q12_launcher, "TPC-H Q12 fused kernel (CUDA)");
}
'''

def _load_cuda() -> object:
    """Compile the CUDA source to a loadable extension."""
    tmpdir = tempfile.mkdtemp(prefix="q12_cuda_")
    src_path = os.path.join(tmpdir, "q12_kernel.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(src_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q12_cuda_ext",
        sources=[src_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

# Load extension once per interpreter
_EXT = _load_cuda()

def _launch_q12(
    shipmode_code: torch.Tensor,
    priority_high: torch.Tensor,
    commit_days: torch.Tensor,
    receipt_days: torch.Tensor,
    ship_days: torch.Tensor,
    receipt_lo: int,
    receipt_hi: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute the fused Q12 kernel.

    Returns
    -------
    group_ids   : torch.Tensor (int64, shape [2]) – identifiers 0 and 1
    high_counts : torch.Tensor (int64, shape [2])
    low_counts  : torch.Tensor (int64, shape [2])
    """
    device = shipmode_code.device
    high_out = torch.zeros(2, dtype=torch.int64, device=device)
    low_out  = torch.zeros(2, dtype=torch.int64, device=device)

    _EXT.q12_launcher(
        shipmode_code,
        priority_high,
        commit_days,
        receipt_days,
        ship_days,
        receipt_lo,
        receipt_hi,
        high_out,
        low_out,
    )

    group_ids = torch.arange(2, dtype=torch.int64, device=device)
    return group_ids, high_out, low_out

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
    Core Q12 computation using the fused CUDA kernel.
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
    SQL:
    SELECT l_shipmode,
           SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH')
                    THEN 1 ELSE 0 END) AS high_line_count,
           SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH')
                    THEN 1 ELSE 0 END) AS low_line_count
    FROM   orders, lineitem
    WHERE  o_orderkey = l_orderkey
           AND l_shipmode IN ('MAIL','SHIP')
           AND l_commitdate < l_receiptdate
           AND l_shipdate < l_commitdate
           AND l_receiptdate >= '1994-01-01'
           AND l_receiptdate <  '1995-01-01'
    GROUP BY l_shipmode
    ORDER BY l_shipmode;

    Description:
        The join on `orders` and `lineitem` is performed with cuDF,
        which runs entirely on the GPU.  After the join, the
        post‑join filtering, ship‑mode encoding, priority flag
        creation, and the two group‑by counts are performed by a
        single fused CUDA kernel.  This eliminates multiple passes
        over the data, reduces intermediate materializations, and
        uses shared‑memory block reductions so that each block
        performs only one atomicAdd per group, dramatically lowering
        contention on H100’s memory subsystem.

    Params:
        shipmode1, shipmode2 – ship‑mode strings to count.
        receipt_lo, receipt_hi – inclusive lower / exclusive upper bound
                                for l_receiptdate.
        priority1, priority2 – order‑priority values considered “high”.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Convert date bounds to epoch days (int32)
    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    # Early filter on shipmode to reduce join size
    li = df_lineitem[df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])]

    # Perform GPU join
    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey")

    # Encode shipmode: 0 = shipmode1, 1 = shipmode2
    shipmode_code = (joined["l_shipmode"] == shipmode2).astype("int32")

    # Encode high‑priority flag
    priority_high = (
        (joined["o_orderpriority"] == priority1)
        | (joined["o_orderpriority"] == priority2)
    ).astype("int32")

    # Convert dates to int32 days since epoch
    l_commit_t = torch.as_tensor(
        ((joined["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    l_receipt_t = torch.as_tensor(
        ((joined["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    l_ship_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    shipmode_t = torch.as_tensor(shipmode_code.to_cupy(), dtype=torch.int32)
    priority_t = torch.as_tensor(priority_high.to_cupy(), dtype=torch.int32)

    # Execute fused kernel
    group_ids, high_counts, low_counts = _query_core(
        shipmode_t,
        priority_t,
        l_commit_t,
        l_receipt_t,
        l_ship_t,
        receipt_lo_days,
        receipt_hi_days,
    )

    # Map group ids back to shipmode strings and ensure alphabetical order
    names = [shipmode1, shipmode2]                 # index matches group id
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_counts.cpu().tolist()
    lc_host = low_counts.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count":  [lc_host[i] for i in ordered],
    })
    return result
