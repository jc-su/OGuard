import os
import tempfile
import pandas as pd
import torch
import cudf
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused filter + revenue computation
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ---------------------------------------------------------------
// Kernel: compute mask (keep rows) and per‑row revenue
// ---------------------------------------------------------------
__global__ void filter_compute_kernel(
    const int64_t* __restrict__ l_orderkey,          // not used, but kept for signature consistency
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate,
    const int32_t* __restrict__ o_orderdate,
    const int32_t* __restrict__ o_shippriority,     // not used inside kernel
    int32_t shipdate_lo,
    int32_t orderdate_hi,
    unsigned char* __restrict__ mask_out,           // torch bool --> uint8_t
    double* __restrict__ rev_out,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bool keep = (l_shipdate[idx] > shipdate_lo) && (o_orderdate[idx] < orderdate_hi);
    mask_out[idx] = static_cast<unsigned char>(keep);
    rev_out[idx] = keep ? l_extendedprice[idx] * (1.0 - l_discount[idx]) : 0.0;
}

// ---------------------------------------------------------------
// Launcher callable from Python
// ---------------------------------------------------------------
void filter_compute_launcher(
    torch::Tensor l_orderkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate,
    torch::Tensor o_orderdate,
    torch::Tensor o_shippriority,
    int32_t shipdate_lo,
    int32_t orderdate_hi,
    torch::Tensor mask_out,
    torch::Tensor rev_out)
{
    // Basic type / device checks
    TORCH_CHECK(l_orderkey.is_cuda(), "l_orderkey must be CUDA");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be CUDA");
    TORCH_CHECK(l_discount.is_cuda(), "l_discount must be CUDA");
    TORCH_CHECK(l_shipdate.is_cuda(), "l_shipdate must be CUDA");
    TORCH_CHECK(o_orderdate.is_cuda(), "o_orderdate must be CUDA");
    TORCH_CHECK(o_shippriority.is_cuda(), "o_shippriority must be CUDA");
    TORCH_CHECK(mask_out.is_cuda(), "mask_out must be CUDA");
    TORCH_CHECK(rev_out.is_cuda(), "rev_out must be CUDA");

    // Ensure contiguous layout
    l_orderkey = l_orderkey.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    l_shipdate = l_shipdate.contiguous();
    o_orderdate = o_orderdate.contiguous();
    o_shippriority = o_shippriority.contiguous();
    mask_out = mask_out.contiguous();
    rev_out = rev_out.contiguous();

    int64_t n = l_orderkey.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    filter_compute_kernel<<<blocks, threads>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate.data_ptr<int32_t>(),
        o_orderdate.data_ptr<int32_t>(),
        o_shippriority.data_ptr<int32_t>(),
        shipdate_lo,
        orderdate_hi,
        reinterpret_cast<unsigned char*>(mask_out.data_ptr<bool>()),
        rev_out.data_ptr<double>(),
        n
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch error: ") + cudaGetErrorString(err));
    }
    // Synchronize to make results visible to Python immediately
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_compute", &filter_compute_launcher,
          "Fuse filter and revenue computation for TPC‑H Q3 (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Helper: compile / load the CUDA extension (once per import)
# ----------------------------------------------------------------------
def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q03_cuda_")
    cu_path = os.path.join(tmpdir, "q03_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q03_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        verbose=False,
        build_directory=build_dir,
    )

# Load the compiled extension (cached across calls)
_EXT = _load_cuda()

# ----------------------------------------------------------------------
# Core GPU compute – replaces the original row‑wise Tensor operations
# ----------------------------------------------------------------------
def _query_core(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU‑accelerated filter + revenue computation (fusion) for TPC‑H Q3.
    The rest of the pipeline (group‑by, aggregation, sorting, top‑N) follows
    the reference implementation using Torch tensor ops.
    """
    # Allocate output buffers
    n = l_orderkey.size(0)
    mask = torch.empty(n, dtype=torch.bool, device=l_orderkey.device)
    rev = torch.empty(n, dtype=torch.float64, device=l_orderkey.device)

    # Launch fused kernel
    _EXT.filter_compute(
        l_orderkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        o_orderdate_days,
        o_shippriority,
        shipdate_lo_days,
        orderdate_hi_days,
        mask,
        rev,
    )

    # ------------------------------------------------------------------
    # Remaining logic identical to reference (group‑by, aggregation, sort)
    # ------------------------------------------------------------------
    lk = l_orderkey[mask]
    od = o_orderdate_days[mask]
    sp = o_shippriority[mask]
    # `rev` already contains revenue per row, masked already (zeros where mask=False)
    rev = rev[mask]

    # Composite key for grouping
    keys = torch.stack([lk.to(torch.int64), od.to(torch.int64), sp.to(torch.int64)], dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    revenue_sum = torch.zeros(unique_keys.shape[0], dtype=rev.dtype, device=rev.device)
    revenue_sum.scatter_add_(0, inverse, rev)

    u_lk = unique_keys[:, 0]
    u_od = unique_keys[:, 1].to(torch.int32)
    u_sp = unique_keys[:, 2].to(torch.int32)

    # ORDER BY revenue DESC, o_orderdate ASC LIMIT top_n
    n_groups = unique_keys.shape[0]
    k = min(top_n, n_groups)

    # Stable secondary sort (orderdate ASC)
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]

    # Primary sort (revenue DESC)
    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    rev_sorted = rev_s[idx_pri][:k]
    lk_sorted = lk_s[idx_pri][:k]
    od_sorted = od_s[idx_pri][:k]
    sp_sorted = sp_s[idx_pri][:k]

    return lk_sorted, rev_sorted, od_sorted, sp_sorted


def run_query(
    tables: dict[str, cudf.DataFrame],
    mktsegment: str = "BUILDING",
    orderdate_hi: str = "1995-03-15",
    shipdate_lo: str = "1995-03-15",
    top_n: int = 10,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate,
               o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < CAST('1995-03-15' AS date)
          AND l_shipdate > CAST('1995-03-15' AS date)
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;
    Description: TPC-H Q3 - Shipping Priority. Joins customer, orders, lineitem;
                 filters by market segment and date predicates; aggregates revenue
                 per (orderkey, orderdate, shippriority); returns top-N by revenue.
    Params:
        mktsegment (str): customer market segment filter.
        orderdate_hi (str): exclusive upper bound on o_orderdate.
        shipdate_lo (str): exclusive lower bound on l_shipdate.
        top_n (int): LIMIT value.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Literal normalization: string dates -> epoch‑day ints
    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    # Filter customers by market segment
    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]

    # Join customer -> orders
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")

    # Join (customer, orders) -> lineitem
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Convert date columns to int32 day‑counts
    o_orderdate_days = ((col["o_orderdate"].astype("int64") // 86400).astype("int32"))
    l_shipdate_days = ((col["l_shipdate"].astype("int64") // 86400).astype("int32"))

    lk_t, rev_t, od_t, sp_t = _query_core(
        torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32),
        shipdate_lo_days,
        orderdate_hi_days,
        int(top_n),
    )

    # Convert orderdate back to datetime64[s] for output
    od_series = cudf.Series(od_t.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_t,
        "revenue": rev_t,
        "o_orderdate": od_series,
        "o_shippriority": sp_t,
    })
