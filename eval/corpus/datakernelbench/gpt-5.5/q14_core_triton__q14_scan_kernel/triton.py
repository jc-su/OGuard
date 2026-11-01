import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "promo_flag_ptr": "*i1",
    "partials_ptr": "*fp64",
    "partial_stride": "i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    price_ptr,
    discount_ptr,
    shipdate_ptr,
    promo_flag_ptr,
    partials_ptr,
    partial_stride,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=in_bounds, other=date_hi)
    keep = in_bounds & (shipdate >= date_lo) & (shipdate < date_hi)

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    promo_flag = tl.load(promo_flag_ptr + offs, mask=keep, other=0)

    revenue = price * (1.0 - discount)
    promo_revenue = tl.where(promo_flag, revenue, 0.0)

    tl.store(partials_ptr + pid, tl.sum(revenue))
    tl.store(partials_ptr + partial_stride + pid, tl.sum(promo_revenue))
