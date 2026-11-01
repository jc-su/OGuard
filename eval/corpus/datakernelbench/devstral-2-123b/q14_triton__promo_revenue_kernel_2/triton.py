import triton
import triton.language as tl

TYPES = {
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "promo_ptr": "*i1",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_total": "*fp64",
    "out_promo": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 16


@triton.jit
def kernel(
    extendedprice_ptr, discount_ptr, shipdate_ptr, promo_ptr,
    date_lo, date_hi, out_total, out_promo, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all needed data in one pass
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    promo_flag = tl.load(promo_ptr + offs, mask=mask, other=False)
    extendedprice = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Combined filter and computation
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    revenue = tl.where(keep, extendedprice * (1.0 - discount), 0.0)
    promo_rev = tl.where(keep & promo_flag, revenue, 0.0)

    # Block-level reductions
    total = tl.sum(revenue)
    promo = tl.sum(promo_rev)

    # Single kernel with shared block memory for better reduction efficiency
    # Use atomic operations only at block level
    tl.atomic_add(out_total, total)
    tl.atomic_add(out_promo, promo)
