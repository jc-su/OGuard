import triton
import triton.language as tl

TYPES = {
    "supp_code_ptr": "*i32",
    "cust_code_ptr": "*i32",
    "shipdate_days_ptr": "*i32",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "year_split_days": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    supp_code_ptr,
    cust_code_ptr,
    shipdate_days_ptr,
    price_ptr,
    discount_ptr,
    date_lo,
    date_hi,
    year_split_days,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    supp = tl.load(supp_code_ptr + offs, mask=mask, other=0)
    cust = tl.load(cust_code_ptr + offs, mask=mask, other=0)
    ship = tl.load(shipdate_days_ptr + offs, mask=mask, other=0)

    pair_keep = ((supp == 0) & (cust == 1)) | ((supp == 1) & (cust == 0))
    date_keep = (ship >= date_lo) & (ship <= date_hi)
    keep = mask & pair_keep & date_keep

    year_offset = tl.where(ship < year_split_days, 0, 1)
    key = supp * 4 + cust * 2 + year_offset

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    volume = price * (1.0 - discount)

    s2 = tl.sum(tl.where(key == 2, volume, 0.0))
    s3 = tl.sum(tl.where(key == 3, volume, 0.0))
    s4 = tl.sum(tl.where(key == 4, volume, 0.0))
    s5 = tl.sum(tl.where(key == 5, volume, 0.0))

    tl.atomic_add(out_ptr + 2, s2)
    tl.atomic_add(out_ptr + 3, s3)
    tl.atomic_add(out_ptr + 4, s4)
    tl.atomic_add(out_ptr + 5, s5)
