import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused filter + revenue aggregation per supplier
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// Kernel: for each lineitem row, if shipdate in range and supplier exists,
// compute revenue and atomically add to per‑supplier total.
__global__ void revenue_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate,
    const int32_t date_lo,
    const int32_t date_hi,
    const int32_t* __restrict__ supp_key_to_idx,
    double* __restrict__ supplier_totals,
    const int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t ship = l_shipdate[idx];
    if (ship < date_lo || ship >= date_hi) return;   // date filter

    int64_t supp = l_suppkey[idx];
    // map supplier key to compact index; -1 means not present in supplier table
    int32_t sup_idx = supp_key_to_idx[supp];
    if (sup_idx < 0) return;                         // no matching supplier

    double revenue = l_extendedprice[idx] * (1.0 - l_discount[idx]);
    atomicAdd(&supplier_totals[sup_idx], revenue);
}

// Launcher: allocate per‑supplier accumulator, invoke kernel, return totals
torch::Tensor revenue_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate,
    int32_t date_lo,
    int32_t date_hi,
    torch::Tensor supp_key_to_idx)
{
    TORCH_CHECK(l_suppkey.dtype() == torch::kInt64, "l_suppkey must be int64");
    TORCH_CHECK(l_extendedprice.dtype() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.dtype() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(l_shipdate.dtype() == torch::kInt32, "l_shipdate must be int32");
    TORCH_CHECK(supp_key_to_idx.dtype() == torch::kInt32, "supp_key_to_idx must be int32");

    // Ensure contiguous tensors on CUDA
    l_suppkey = l_suppkey.contiguous().to(torch::kCUDA);
    l_extendedprice = l_extendedprice.contiguous().to(torch::kCUDA);
    l_discount = l_discount.contiguous().to(torch::kCUDA);
    l_shipdate = l_shipdate.contiguous().to(torch::kCUDA);
    supp_key_to_idx = supp_key_to_idx.contiguous().to(torch::kCUDA);

    int64_t n = l_suppkey.numel();
    int64_t n_sup = supp_key_to_idx.size(0) - 1; // mapping size = max_key+1, last entry unused for count
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(l_suppkey.device());
    torch::Tensor totals = torch::zeros({n_sup}, options);

    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    revenue_kernel<<<blocks, threads>>>(
        l_suppkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate.data_ptr<int32_t>(),
        date_lo,
        date_hi,
        supp_key_to_idx.data_ptr<int32_t>(),
        totals.data_ptr<double>(),
        n);

    // synchronize to surface any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch error: ") + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return totals;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("revenue_launcher", &revenue_launcher, "TPCH Q15 revenue per supplier (CUDA)");
}
'''

def _load_cuda() -> object:
    """Compile the embedded CUDA source and return the loaded extension module."""
    tmpdir = tempfile.mkdtemp(prefix="q15_cuda_")
    src_path = os.path.join(tmpdir, "q15_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(src_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q15_cuda_ext",
        sources=[src_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        verbose=False,
        with_cuda=True,
        build_directory=build_dir,
    )

# Load (or re‑load) the CUDA extension once per process
_EXT = _load_cuda()

# ----------------------------------------------------------------------
# Core query logic – uses the fused CUDA kernel for the heavy scan/agg
# ----------------------------------------------------------------------
def _query_core(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute the hot path of TPCH Q15 on GPU.

    Parameters
    ----------
    l_suppkey : torch.Tensor[int64]
        Supplier key from lineitem.
    l_extendedprice : torch.Tensor[float64]
        Extended price from lineitem.
    l_discount : torch.Tensor[float64]
        Discount from lineitem.
    l_shipdate_days : torch.Tensor[int32]
        Shipdate expressed as days since epoch.
    s_suppkey : torch.Tensor[int64]
        Supplier keys (original table order).
    shipdate_lo_days, shipdate_hi_days : int
        Inclusive / exclusive shipdate bounds, expressed as days since epoch.

    Returns
    -------
    tuple of three tensors:
        sel_idx      – indices (into the original supplier table) of the winning supplier(s)
        sel_keys     – supplier keys of the winners (ordered)
        sel_revenue  – total revenue of each winner (ordered)
    """
    device = l_suppkey.device

    # ------------------------------------------------------------------
    # Build a dense map from supplier key -> compact index (0 .. n_sup‑1)
    # ------------------------------------------------------------------
    n_sup = s_suppkey.numel()
    max_key = int(s_suppkey.max().item())
    map_size = max_key + 1                     # include max_key itself
    # initialise with -1 (meaning “no supplier”)
    supp_map = torch.full((map_size,), -1, dtype=torch.int32, device=device)
    idx = torch.arange(n_sup, dtype=torch.int32, device=device)
    supp_map.scatter_(0, s_suppkey.to(torch.int64), idx)

    # ------------------------------------------------------------------
    # Run fused CUDA kernel: per‑supplier revenue totals
    # ------------------------------------------------------------------
    totals = _EXT.revenue_launcher(
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        supp_map,
    )  # returns DoubleTensor on GPU, length = n_sup

    # ------------------------------------------------------------------
    # Identify suppliers with maximal revenue (excluding those with zero revenue)
    # ------------------------------------------------------------------
    present = totals != 0.0
    if present.any():
        max_rev = totals[present].max()
        sel_mask = present & (totals == max_rev)
    else:
        # No revenue at all – treat all as zero‑revenue winners (matches reference logic)
        max_rev = torch.tensor(0.0, dtype=torch.float64, device=device)
        sel_mask = torch.ones_like(totals, dtype=torch.bool, device=device)

    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)          # indices into supplier order
    sel_keys = s_suppkey[sel_idx]

    # Order final result by supplier key as required by SQL
    order = torch.argsort(sel_keys)
    sel_idx = sel_idx[order]

    return sel_idx, s_suppkey[sel_idx], totals[sel_idx]

# ----------------------------------------------------------------------
# Public API – unchanged signature
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1996-01-01",
    shipdate_hi: str = "1996-04-01",
) -> cudf.DataFrame:
    """
    SQL:
      WITH revenue AS (
        SELECT l_suppkey AS supplier_no,
               SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
        GROUP BY l_suppkey
      )
      SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
      FROM supplier, revenue
      WHERE s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
      ORDER BY s_suppkey;
    Description: TPC‑H Q15 – find supplier(s) with the top revenue in a quarter.
                 The heavy scan/aggregation is done in a single fused CUDA kernel,
                 avoiding multiple cuDF passes and temporary buffers.
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    # ------------------------------------------------------------------
    # Pull columns from cuDF tables and move them to CuTorch tensors
    # ------------------------------------------------------------------
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    # Convert date bounds to integer days since epoch
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Lineitem columns
    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64, device="cuda")
    l_price_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64, device="cuda")
    l_disc_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64, device="cuda")
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
        device="cuda",
    )

    # Supplier columns (keys only needed for GPU work)
    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64, device="cuda")

    # ------------------------------------------------------------------
    # Execute core computation on GPU
    # ------------------------------------------------------------------
    sel_idx_t, sel_keys_t, sel_rev_t = _query_core(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        s_suppkey_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # ------------------------------------------------------------------
    # Gather string columns from the original cuDF supplier table using the indices
    # ------------------------------------------------------------------
    # Convert torch index tensor to cuDF series (cuDF accepts torch tensors directly)
    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    # Build the final result DataFrame
    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_t),
    })
