import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "promo_ptr": "*i1",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr, discount_ptr, shipdate_ptr, promo_ptr,
    date_lo, date_hi, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load and filter by shipdate
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = (shipdate >= date_lo) & (shipdate < date_hi)
    keep_mask = mask & keep
    
    # Conditionally load expensive columns (hardware skips fetch if keep_mask is false for the warp)
    price = tl.load(price_ptr + offs, mask=keep_mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep_mask, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=keep_mask, other=0)
    
    # Compute revenue and mask out invalid lanes
    revenue = price * (1.0 - discount)
    revenue = tl.where(keep_mask, revenue, 0.0)
    
    # Compute promotional revenue
    promo_revenue = tl.where(promo != 0, revenue, 0.0)
    
    # Compute block sums
    total_rev = tl.sum(revenue)
    total_promo = tl.sum(promo_revenue)
    
    # Accumulate globally
    tl.atomic_add(out_ptr, total_promo)
    tl.atomic_add(out_ptr + 1, total_rev)
