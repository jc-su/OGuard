import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd


# All Triton kernels must be in a separate on-disk file (validator requirement)
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
    price_ptr, discount_ptr, quantity_ptr, shipdate_ptr,
    date_lo, date_hi, thresh_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused filter + project + aggregate kernel for TPC-H Q6.
    
    Applies all predicates in registers (no extra HBM traffic) and performs
    conditional project + sum in a single pass. Uses atomic_add for global reduction.
    """
    # Load threshold values from device tensor (fp64 to match column precision)
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi = tl.load(thresh_ptr + 2)
    
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load filter columns (cheap columns: date int32, discount fp64, quantity fp64)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=qty_hi)
    
    # Predicate evaluation in registers (no additional HBM reads)
    keep = (
        mask
        & (shipdate >= date_lo) & (shipdate < date_hi)
        & (discount >= disc_lo) & (discount <= disc_hi)
        & (quantity < qty_hi)
    )
    
    # Conditional load of expensive column (price fp64) only where needed
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    
    # Project and aggregate: sum(price * discount) for matching rows
    # Using tl.where to zero out non-matching rows before sum
    contribution = tl.where(keep, price * discount, 0.0)
    block_sum = tl.sum(contribution)
    
    # Atomic add for global reduction across all blocks
    tl.atomic_add(out_ptr, block_sum)
'''


def _load_kernels():
    """Load Triton kernels from on-disk file (required by validator)."""
    tmpdir = tempfile.mkdtemp(prefix="q6_triton_")
    path = os.path.join(tmpdir, "_q6_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q6_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pre-load kernels at module import time
_K = _load_kernels()


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

    Description: Computes the forecast revenue increase from eliminating discounts in a given
    range for a year. Uses a fused Triton kernel for filter + project + aggregate in a single
    pass over GPU memory, reducing bandwidth compared to the reference PyTorch plan which
    creates masks and performs multiple passes.

    Params:
        tables (dict[str, cudf.DataFrame]): Dictionary containing the 'lineitem' table.
        shipdate_lo (str): inclusive lower bound ship date.
        shipdate_hi (str): exclusive upper bound ship date.
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
    """
    df_lineitem = tables["lineitem"]

    # Convert date strings to int32 epoch days (same as reference for correctness)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Convert cuDF columns to GPU tensors (float64 for aggregates, int32 for dates)
    l_extendedprice_t = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        df_lineitem["l_discount"].to_cupy(), dtype=torch.float64
    )
    l_quantity_t = torch.as_tensor(
        df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64
    )
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    n_elements = l_extendedprice_t.numel()

    # Output tensor (fp64 for precision matching reference)
    out = torch.zeros(1, dtype=torch.float64, device=l_extendedprice_t.device)

    # Pack scalar thresholds as fp64 device tensor (critical for precision)
    thresh = torch.tensor(
        [discount_lo, discount_hi, quantity_hi],
        dtype=torch.float64,
        device=l_extendedprice_t.device,
    )

    # Launch fused kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _K._q6_kernel[grid](
        l_extendedprice_t,
        l_discount_t,
        l_quantity_t,
        l_shipdate_days_t,
        shipdate_lo_days,
        shipdate_hi_days,
        thresh,
        out,
        n_elements,
    )

    return float(out.item())
