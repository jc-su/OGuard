import triton
import triton.language as tl

TYPES = {
    "shipdate_ptr": "*i32",
    "discount_ptr": "*fp64",
    "price_ptr": "*fp64",
    "promo_ptr": "*i1",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_total_ptr": "*fp64",
    "out_promo_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    shipdate_ptr, discount_ptr, price_ptr, promo_ptr,
    date_lo, date_hi,
    out_total_ptr, out_promo_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns - all at once for coalesced access
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=mask, other=0)

    # Filter predicate: shipdate in range [date_lo, date_hi)
    keep = (shipdate >= date_lo) & (shipdate < date_hi)

    # Project: revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)

    # Apply filter: zero out rows outside date range
    revenue_filtered = tl.where(keep, revenue, 0.0)

    # Aggregate: total sum
    total = tl.sum(revenue_filtered)
    tl.atomic_add(out_total_ptr, total)

    # Aggregate: promo sum (only where promo flag is True AND date filter passes)
    promo_revenue = tl.where(keep & promo, revenue, 0.0)
    promo_sum = tl.sum(promo_revenue)
    tl.atomic_add(out_promo_ptr, promo_sum)
