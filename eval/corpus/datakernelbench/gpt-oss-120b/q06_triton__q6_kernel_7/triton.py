import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "quantity_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "thresh_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr,         # fp64 *
    discount_ptr,      # fp64 *
    quantity_ptr,      # fp64 *
    shipdate_ptr,      # int32 *
    date_lo,           # int32 scalar
    date_hi,           # int32 scalar
    thresh_ptr,        # fp64 *  (discount_lo, discount_hi, quantity_hi)
    out_ptr,           # fp64 *
    n_elements,        # int64 scalar
    BLOCK_SIZE: tl.constexpr,
):
    # load scalar thresholds (they reside in a small device array)
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi  = tl.load(thresh_ptr + 2)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements  # guard against OOB

    # Load columns (int32 for shipdate, fp64 for others)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=qty_hi)

    # Predicate evaluation – kept entirely in registers
    keep = (
        mask
        & (shipdate >= date_lo) & (shipdate < date_hi)
        & (discount >= disc_lo) & (discount <= disc_hi)
        & (quantity < qty_hi)
    )

    # Conditional load of the expensive column only for rows that survive
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)

    # Contribution to revenue = price * discount for kept rows
    contrib = tl.where(keep, price * discount, 0.0)

    # Reduce within the block and atomically add to global accumulator
    tl.atomic_add(out_ptr, tl.sum(contrib))
