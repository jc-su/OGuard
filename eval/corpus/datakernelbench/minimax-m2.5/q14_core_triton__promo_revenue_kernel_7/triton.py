import triton
import triton.language as tl

TYPES = {
    "shipdate_ptr": "*i32",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "promo_ptr": "*i1",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_total_ptr": "*fp64",
    "out_promo_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    shipdate_ptr, price_ptr, discount_ptr, promo_ptr,
    date_lo, date_hi, out_total_ptr, out_promo_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate with a sentinel value that fails the date predicate
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    
    # Apply date filter: shipdate_lo <= shipdate < shipdate_hi
    date_keep = (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Load price and discount only for rows that pass date filter
    # This conditional load is a key optimization for low-selectivity filters
    price = tl.load(price_ptr + offs, mask=date_keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=date_keep, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=date_keep, other=0.0)
    
    # Compute revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)
    
    # Accumulate total revenue
    block_total = tl.sum(revenue)
    tl.atomic_add(out_total_ptr, block_total)
    
    # Accumulate promo revenue (where promo flag is True)
    promo_revenue = tl.where(promo, revenue, 0.0)
    block_promo = tl.sum(promo_revenue)
    tl.atomic_add(out_promo_ptr, block_promo)
