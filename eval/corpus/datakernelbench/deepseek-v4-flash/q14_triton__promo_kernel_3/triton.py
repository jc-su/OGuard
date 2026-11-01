import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "promo_ptr": "*i1",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_total_ptr": "*fp64",
    "out_promo_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 2048,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr, discount_ptr, shipdate_ptr, promo_ptr,
    date_lo, date_hi, out_total_ptr, out_promo_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=mask, other=0)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    revenue = price * (1.0 - discount)
    total_block = tl.sum(tl.where(keep, revenue, 0.0))
    promo_block = tl.sum(tl.where(keep & (promo > 0), revenue, 0.0))
    tl.atomic_add(out_total_ptr, total_block)
    tl.atomic_add(out_promo_ptr, promo_block)
