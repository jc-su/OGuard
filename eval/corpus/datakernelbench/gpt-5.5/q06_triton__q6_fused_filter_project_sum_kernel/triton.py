import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "quantity_ptr": "*fp64",
    "shipdate_days_ptr": "*i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "thresholds_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 8


@triton.jit
def kernel(
    price_ptr,
    discount_ptr,
    quantity_ptr,
    shipdate_days_ptr,
    date_lo,
    date_hi,
    thresholds_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    disc_lo = tl.load(thresholds_ptr + 0)
    disc_hi = tl.load(thresholds_ptr + 1)
    qty_hi = tl.load(thresholds_ptr + 2)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    shipdate = tl.load(shipdate_days_ptr + offs, mask=in_bounds, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=in_bounds, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=in_bounds, other=qty_hi)

    keep = (
        in_bounds
        & (shipdate >= date_lo)
        & (shipdate < date_hi)
        & (discount >= disc_lo)
        & (discount <= disc_hi)
        & (quantity < qty_hi)
    )

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    contrib = tl.where(keep, price * discount, 0.0)
    block_sum = tl.sum(contrib, axis=0)

    tl.atomic_add(out_ptr, block_sum, sem="relaxed")
