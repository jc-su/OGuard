import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "promo_flag_ptr": "*i1",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_promo": "*fp64",
    "out_total": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr, discount_ptr, shipdate_ptr, promo_flag_ptr,
    date_lo, date_hi, out_promo, out_total, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    promo = tl.load(promo_flag_ptr + offs, mask=keep, other=0)

    revenue = price * (1.0 - discount)
    block_total = tl.sum(tl.where(keep, revenue, 0.0))
    block_promo = tl.sum(tl.where((keep == 1) & (promo == 1), revenue, 0.0))

    if block_total != 0.0:
        tl.atomic_add(out_total, block_total)
        tl.atomic_add(out_promo, block_promo)
