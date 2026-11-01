import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "shipdate_days_ptr": "*i32",
    "promo_ptr": "*i1",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    price_ptr,
    discount_ptr,
    shipdate_days_ptr,
    promo_ptr,
    date_lo,
    date_hi,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    shipdate = tl.load(shipdate_days_ptr + offs, mask=in_bounds, other=date_hi)
    keep = in_bounds & (shipdate >= date_lo) & (shipdate < date_hi)

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=keep, other=0)

    revenue = price * (1.0 - discount)
    total_revenue = tl.sum(tl.where(keep, revenue, 0.0))
    promo_revenue = tl.sum(tl.where(keep & promo, revenue, 0.0))

    tl.atomic_add(out_ptr, total_revenue, sem="relaxed")
    tl.atomic_add(out_ptr + 1, promo_revenue, sem="relaxed")
