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

CONSTANTS = {
    "BLOCK_SIZE": 8192,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr: tl.tensor,
    discount_ptr: tl.tensor,
    quantity_ptr: tl.tensor,
    shipdate_ptr: tl.tensor,
    date_lo: tl.int32,
    date_hi: tl.int32,
    thresh_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi = tl.load(thresh_ptr + 2)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=qty_hi)
    keep = (
        mask
        & (shipdate >= date_lo) & (shipdate < date_hi)
        & (discount >= disc_lo) & (discount <= disc_hi)
        & (quantity < qty_hi)
    )
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, price * discount, 0.0)))
