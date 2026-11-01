import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton  # needed for triton.cdiv in grid calculation

# -------------------------------------------------
# Triton kernel source (must be written to disk)
# -------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune across a range of block sizes and warps for H100
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=wm, num_stages=st)
        for bs in [1024, 2048, 4096, 8192]
        for wm in [4, 8]
        for st in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],  # ensure clean reduction on each launch
)
@triton.jit
def _q6_kernel(
    price_ptr,         # fp64 *
    discount_ptr,      # fp64 *
    quantity_ptr,      # fp64 *
    shipdate_ptr,      # int32 *
    date_lo,           # int32 scalar
    date_hi,           # int32 scalar
    thresh_ptr,        # fp64 *  (discount_lo, discount_hi, quantity_hi)
    out_ptr,           # fp64 *
    n_elements,        # int64 scalar
    BLOCK_SIZE: tl.constexpr,
):
    # load scalar thresholds (they reside in a small device array)
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi  = tl.load(thresh_ptr + 2)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements  # guard against OOB

    # Load columns (int32 for shipdate, fp64 for others)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=qty_hi)

    # Predicate evaluation – kept entirely in registers
    keep = (
        mask
        & (shipdate >= date_lo) & (shipdate < date_hi)
        & (discount >= disc_lo) & (discount <= disc_hi)
        & (quantity < qty_hi)
    )

    # Conditional load of the expensive column only for rows that survive
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)

    # Contribution to revenue = price * discount for kept rows
    contrib = tl.where(keep, price * discount, 0.0)

    # Reduce within the block and atomically add to global accumulator
    tl.atomic_add(out_ptr, tl.sum(contrib))
'''

def _load_kernels():
    """Write the kernel source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q6_triton_")
    path = os.path.join(tmpdir, "_q6_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q6_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load the Triton kernels at import time
_K = _load_kernels()

def _launch_q6(
    price: torch.Tensor,
    discount: torch.Tensor,
    quantity: torch.Tensor,
    shipdate_days: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    discount_lo: float,
    discount_hi: float,
    quantity_hi: float,
) -> torch.Tensor:
    """
    Launch the fused Triton kernel that filters, projects and aggregates
    in a single pass. Returns a 0‑dim tensor containing the revenue sum.
    """
    n = price.numel()
    # Output buffer (fp64) – initialized to zero for each launch
    out = torch.zeros(1, dtype=torch.float64, device=price.device)

    # Pack scalar thresholds into a tiny fp64 tensor on the device
    thresh = torch.tensor(
        [discount_lo, discount_hi, quantity_hi],
        dtype=torch.float64,
        device=price.device,
    )

    # Compute grid size based on the autotuned BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    # Triton kernel launch
    _K._q6_kernel[grid](
        price,
        discount,
        quantity,
        shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        thresh,
        out,
        n,
    )
    return out.squeeze(0)

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
    Core query implementation – a thin wrapper around the fused Triton launch.
    All heavy lifting (filter + multiply + sum) happens inside the kernel.
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

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
    discount_lo: float = 0.05,
    discount_hi: float = 0.07,
    quantity_hi: float = 24.0,
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= CAST('1994-01-01' AS date)
          AND l_shipdate <  CAST('1995-01-01' AS date)
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24;

    Description:
        The query scans the `lineitem` table, applying three predicates and
        computing `l_extendedprice * l_discount` for the qualifying rows.
        The reference implementation performed separate Torch operations,
        incurring multiple memory reads.  This version fuses filtering,
        projection and the global reduction into a single Triton kernel,
        reducing HBM traffic and latency on the H100 (memory‑bound workload).

    Params:
        shipdate_lo (str): inclusive lower bound for `l_shipdate`.
        shipdate_hi (str): exclusive upper bound for `l_shipdate`.
        discount_lo (float): inclusive lower bound for `l_discount`.
        discount_hi (float): inclusive upper bound for `l_discount`.
        quantity_hi (float): exclusive upper bound for `l_quantity`.

    Returns:
        float: revenue (sum of extendedprice * discount) for the filtered rows.
    """
    df = tables["lineitem"]

    # Convert dates to integer days since epoch (int32) – matches reference logic
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Transfer columns to GPU tensors (fp64 for monetary columns, fp64 for discounts/qty)
    l_extendedprice_t = torch.as_tensor(
        df["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        df["l_discount"].to_cupy(), dtype=torch.float64
    )
    l_quantity_t = torch.as_tensor(
        df["l_quantity"].to_cupy(), dtype=torch.float64
    )
    l_shipdate_days_t = torch.as_tensor(
        ((df["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    # Execute the fused kernel
    result_tensor = _query_core(
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
    # Convert the 0‑dim tensor to a Python float
    return float(result_tensor.item())
