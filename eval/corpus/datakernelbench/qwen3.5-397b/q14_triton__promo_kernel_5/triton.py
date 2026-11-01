import triton
import triton.language as tl

TYPES = {
    "shipdate_ptr": "*i32",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "promo_ptr": "*i1",
    "out_total_ptr": "*fp64",
    "out_promo_ptr": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 4


@triton.jit
def kernel(
    shipdate_ptr,
    price_ptr,
    discount_ptr,
    promo_ptr,
    out_total_ptr,
    out_promo_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load columns
    # shipdate is int32 days
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    # price/discount are float64
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    # promo is bool
    promo = tl.load(promo_ptr + offs, mask=mask, other=False)
    
    # Date filter: shipdate_lo <= shipdate < shipdate_hi
    keep_date = (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Revenue calculation: extendedprice * (1 - discount)
    revenue = price * (1.0 - discount)
    
    # Conditional accumulation
    # If keep_date is false, contribution is 0.0
    total_contrib = tl.where(keep_date, revenue, 0.0)
    # If keep_date AND promo are true, contribution is revenue, else 0.0
    promo_contrib = tl.where(keep_date & promo, revenue, 0.0)
    
    # Block-level reduction
    total_sum = tl.sum(total_contrib)
    promo_sum = tl.sum(promo_contrib)
    
    # Global accumulation
    tl.atomic_add(out_total_ptr, total_sum)
    tl.atomic_add(out_promo_ptr, promo_sum)
