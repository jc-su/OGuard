import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused filter + group‑by aggregation for TPCH Q1
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 1024
#define MAX_GROUPS (256*256)   // rf_code * 256 + ls_code

__global__ void q1_kernel(
    const double* __restrict__ qty,
    const double* __restrict__ price,
    const double* __restrict__ disc,
    const double* __restrict__ tax,
    const int*    __restrict__ rf,
    const int*    __restrict__ ls,
    const int*    __restrict__ shipdate,
    const int      shipdate_hi,
    double* __restrict__ sum_qty,
    double* __restrict__ sum_price,
    double* __restrict__ sum_disc_price,
    double* __restrict__ sum_charge,
    double* __restrict__ sum_disc,
    unsigned long long* __restrict__ cnt,
    const int64_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (shipdate[idx] <= shipdate_hi) {
        int key = rf[idx] * 256 + ls[idx];   // encodes group

        double q   = qty[idx];
        double p   = price[idx];
        double d   = disc[idx];
        double t   = tax[idx];
        double dp  = p * (1.0 - d);          // discounted price
        double ch  = dp * (1.0 + t);         // charge

        atomicAdd(&sum_qty[key],        q);
        atomicAdd(&sum_price[key],      p);
        atomicAdd(&sum_disc_price[key], dp);
        atomicAdd(&sum_charge[key],     ch);
        atomicAdd(&sum_disc[key],       d);
        atomicAdd(&cnt[key], 1ULL);
    }
}

// launcher returning a list of tensors: 5 sums (double), 1 count (int64)
std::vector<torch::Tensor> q1_launcher(
    torch::Tensor qty,
    torch::Tensor price,
    torch::Tensor disc,
    torch::Tensor tax,
    torch::Tensor rf,
    torch::Tensor ls,
    torch::Tensor shipdate,
    const int32_t shipdate_hi)
{
    TORCH_CHECK(qty.is_cuda() && price.is_cuda() && disc.is_cuda() &&
                tax.is_cuda() && rf.is_cuda() && ls.is_cuda() && shipdate.is_cuda(),
                "All inputs must be CUDA tensors");
    const int64_t N = qty.numel();

    auto opt_f64 = torch::TensorOptions().dtype(torch::kFloat64).device(qty.device());
    auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(qty.device());

    // allocate per‑group accumulator buffers
    torch::Tensor sum_qty        = torch::zeros({MAX_GROUPS}, opt_f64);
    torch::Tensor sum_price      = torch::zeros({MAX_GROUPS}, opt_f64);
    torch::Tensor sum_disc_price = torch::zeros({MAX_GROUPS}, opt_f64);
    torch::Tensor sum_charge     = torch::zeros({MAX_GROUPS}, opt_f64);
    torch::Tensor sum_disc       = torch::zeros({MAX_GROUPS}, opt_f64);
    torch::Tensor cnt            = torch::zeros({MAX_GROUPS}, opt_i64);

    const int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q1_kernel<<<blocks, BLOCK_SIZE>>>(
        qty.data_ptr<double>(),
        price.data_ptr<double>(),
        disc.data_ptr<double>(),
        tax.data_ptr<double>(),
        rf.data_ptr<int>(),
        ls.data_ptr<int>(),
        shipdate.data_ptr<int>(),
        shipdate_hi,
        sum_qty.data_ptr<double>(),
        sum_price.data_ptr<double>(),
        sum_disc_price.data_ptr<double>(),
        sum_charge.data_ptr<double>(),
        sum_disc.data_ptr<double>(),
        reinterpret_cast<unsigned long long*>(cnt.data_ptr<int64_t>()), // atomicAdd works on ULL
        N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return {sum_qty, sum_price, sum_disc_price, sum_charge, sum_disc, cnt};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q1_launcher", &q1_launcher, "Fused Q1 aggregation (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Load/compile the CUDA extension exactly once at import time
# ----------------------------------------------------------------------
def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="tpch_q1_cuda_")
    cu_path = os.path.join(tmpdir, "q1_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="tpch_q1_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

_EXT = _load_cuda()

# ----------------------------------------------------------------------
# Private launcher that calls the compiled CUDA function
# ----------------------------------------------------------------------
def _launch_q1(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calls the fused CUDA kernel that computes per‑group sums and counts.
    Returns six tensors on the same device:
        sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc, count
    """
    # Ensure contiguous inputs (kernel expects contiguous memory)
    qty = l_quantity.contiguous()
    price = l_extendedprice.contiguous()
    disc = l_discount.contiguous()
    tax = l_tax.contiguous()
    rf = l_returnflag_code.contiguous()
    ls = l_linestatus_code.contiguous()
    ship = l_shipdate_days.contiguous()

    # CUDA launcher returns a list of tensors in the order defined above
    sum_qty, sum_price, sum_disc_price, sum_charge, sum_disc, cnt = _EXT.q1_launcher(
        qty, price, disc, tax, rf, ls, ship, shipdate_hi_days
    )
    return sum_qty, sum_price, sum_disc_price, sum_charge, sum_disc, cnt

# ----------------------------------------------------------------------
# Public core function – signature must match the reference plan
# ----------------------------------------------------------------------
def _query_core(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """
    GPU‑accelerated aggregation for TPCH Q1.
    Performs a single fused CUDA pass that filters on shipdate,
    groups by (l_returnflag, l_linestatus) and computes all required aggregates.
    The result tensors are ordered and typed exactly like the reference implementation.
    """
    # 1. Run the fused kernel → per‑group buffers (size MAX_GROUPS)
    sum_qty, sum_price, sum_disc_price, sum_charge, sum_disc, cnt = _launch_q1(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        shipdate_hi_days,
    )

    # 2. Keep only groups that actually occurred (cnt != 0)
    mask = cnt != 0
    keys = torch.nonzero(mask, as_tuple=False).squeeze()               # shape (n_groups,)
    # Gather the per‑group values
    sum_qty = sum_qty[keys]
    sum_price = sum_price[keys]
    sum_disc_price = sum_disc_price[keys]
    sum_charge = sum_charge[keys]
    sum_disc = sum_disc[keys]
    cnt = cnt[keys]

    # 3. Decode the packed group key back to the original categorical codes
    rf_out = (keys // 256).to(torch.int64)
    ls_out = (keys % 256).to(torch.int64)

    # 4. Compute averages (using double precision)
    cnt_f = cnt.to(torch.float64)
    avg_qty = sum_qty / cnt_f
    avg_price = sum_price / cnt_f
    avg_disc = sum_disc / cnt_f

    # 5. Order rows by (l_returnflag, l_linestatus) → same as sorting by packed key
    order = torch.argsort(keys)

    return (
        rf_out[order],
        ls_out[order],
        sum_qty[order],
        sum_price[order],
        sum_disc_price[order],
        sum_charge[order],
        avg_qty[order],
        avg_price[order],
        avg_disc[order],
        cnt[order],
    )

# ----------------------------------------------------------------------
# run_query – unchanged from the reference plan (only imports may be added)
# ----------------------------------------------------------------------
def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
    Description: TPC-H Q1 pricing summary report aggregated by return flag and line status.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]

    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode string columns l_returnflag, l_linestatus to integer codes (single-char strings).
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    # Use categorical codes for encoding
    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    l_returnflag_code_t = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy())
    l_linestatus_code_t = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy())

    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    (
        rf_out, ls_out, sum_qty, sum_base_price, sum_disc_price,
        sum_charge, avg_qty, avg_price, avg_disc, count_order,
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
    )

    # Decode category codes back to strings
    rf_out_host = rf_out.cpu().tolist()
    ls_out_host = ls_out.cpu().tolist()
    rf_strs = [rf_categories[i] for i in rf_out_host]
    ls_strs = [ls_categories[i] for i in ls_out_host]

    return cudf.DataFrame({
        "l_returnflag": cudf.Series(rf_strs),
        "l_linestatus": cudf.Series(ls_strs),
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": sum_disc_price,
        "sum_charge": sum_charge,
        "avg_qty": avg_qty,
        "avg_price": avg_price,
        "avg_disc": avg_disc,
        "count_order": count_order,
    })
