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

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr, discount_ptr, quantity_ptr, shipdate_ptr,
    date_lo, date_hi, thresh_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Load thresholds (fp64 for precision)
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi = tl.load(thresh_ptr + 2)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all filter columns
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=999.0)  # Safe upper bound

    # Apply all predicates
    keep = (
        mask &
        (shipdate >= date_lo) &
        (shipdate < date_hi) &
        (discount >= disc_lo) &
        (discount <= disc_hi) &
        (quantity < qty_hi)
    )

    # Conditional compute of contribution
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    revenue = tl.where(keep, price * discount, 0.0)

    # Reduce and atomic add
    block_sum = tl.sum(revenue, axis=0)
    tl.atomic_add(out_ptr, block_sum)
