import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source: fused filter → project → aggregate
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q6_kernel(
    price_ptr,
    discount_ptr,
    quantity_ptr,
    shipdate_ptr,
    date_lo,
    date_hi,
    thresh_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Load scalar thresholds (float64)
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi = tl.load(thresh_ptr + 2)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns (int32 shipdate, float64 discount, float64 quantity)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=qty_hi)

    # Predicate evaluation
    keep = (
        mask
        & (shipdate >= date_lo)
        & (shipdate < date_hi)
        & (discount >= disc_lo)
        & (discount <= disc_hi)
        & (quantity < qty_hi)
    )

    # Project: price * discount for kept rows
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    contrib = tl.where(keep, price * discount, 0.0)

    # Block-level reduction
    block_sum = tl.sum(contrib)

    # Atomic add to global accumulator
    tl.atomic_add(out_ptr, block_sum)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q6_triton_")
    path = os.path.join(tmpdir, "_q6_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q6_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

# ----------------------------------------------------------------------
# Launcher: prepares grid, thresholds tensor, output tensor
# ----------------------------------------------------------------------
def _launch_q6(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_quantity: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    discount_lo: float,
    discount_hi: float,
    quantity_hi: float,
) -> torch.Tensor:
    """
    Fused Triton kernel for the hot path:
    filter rows by shipdate, discount, quantity and compute
    sum(l_extendedprice * l_discount).
    """
    n = l_extendedprice.numel()
    # Scalar output (float64 to match precision of aggregates)
    out = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)

    # Pack scalar thresholds into a float64 device tensor
    thresh = torch.tensor(
        [discount_lo, discount_hi, quantity_hi],
        dtype=torch.float64,
        device=l_extendedprice.device,
    )

    # Launch grid based on autotuned block size
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q6_kernel[grid](
        l_extendedprice,
        l_discount,
        l_quantity,
        l_shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        thresh,
        out,
        n,
    )
    return out.squeeze(0)


# ----------------------------------------------------------------------
# Core compute (hot path) – replaces eager PyTorch with fused Triton
# ----------------------------------------------------------------------
def _query_core(
    l_shipdate_days: torch.Tensor,
    l_discount: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    discount_lo: float,
    discount_hi: float,
    quantity_hi: float,
) -> torch.Tensor:
    """
    Applies the TPC‑H Q6 filter and aggregation using a single fused Triton kernel.
    Returns a scalar tensor with the sum.
    """
    return _launch_q6(
        l_extendedprice,
        l_discount,
        l_quantity,
        l_shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
    )


# ----------------------------------------------------------------------
# Reference run_query – unchanged from the reference Torch plan
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
    discount_lo: float = 0.05,
    discount_hi: float = 0.07,
    quantity_hi: float = 24.0,
) -> float:
    """
    SQL: SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem
         WHERE l_shipdate >= CAST('1994-01-01' AS date) AND l_shipdate < CAST('1995-01-01' AS date)
         AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;
    Description: Computes the forecast revenue increase from eliminating discounts in a given range for a year.
    Params:
        shipdate_lo (str): inclusive lower bound ship date.
        shipdate_hi (str): exclusive upper bound ship date.
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
    """
    df_lineitem = tables["lineitem"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(
        l_shipdate_days_t,
        l_discount_t,
        l_quantity_t,
        l_extendedprice_t,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
    )
    return float(res.item())
