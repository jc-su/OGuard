import triton
import triton.language as tl

TYPES = {
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "promo_ptr": "*i1",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 4


@triton.jit
def kernel(
    extendedprice_ptr,
    discount_ptr,
    shipdate_ptr,
    promo_ptr,
    date_lo,
    date_hi,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q14 promo revenue calculation.
    
    Single pass through all columns:
    1. Load shipdate, apply date filter (cheap, early reject)
    2. Load extendedprice, discount, compute revenue = price * (1 - discount)
    3. Load promo flag, compute promo revenue where date AND promo
    4. Block-level sum, atomic add to global accumulators
    
    Memory-bound optimization: fusion reduces HBM passes from 4+ (eager) to 1.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate first for date filter (int32 epoch days)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    
    # Date filter: shipdate_lo <= shipdate < shipdate_hi
    date_keep = (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Load remaining columns (all rows, but revenue zeroed for non-matching dates)
    extendedprice = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    is_promo = tl.load(promo_ptr + offs, mask=mask, other=False)
    
    # Revenue calculation: extendedprice * (1 - discount)
    revenue = extendedprice * (1.0 - discount)
    
    # Total revenue (date-filtered)
    filtered_revenue = tl.where(date_keep, revenue, 0.0)
    
    # Promo revenue (date-filtered AND promo flag)
    promo_revenue = tl.where(date_keep & is_promo, revenue, 0.0)
    
    # Block-level sums
    total_block = tl.sum(filtered_revenue)
    promo_block = tl.sum(promo_revenue)
    
    # Atomic add to global output [total, promo]
    # Note: out_ptr + 0 and out_ptr + 1 are pointer offsets, not mask args
    tl.atomic_add(out_ptr + 0, total_block)
    tl.atomic_add(out_ptr + 1, promo_block)
