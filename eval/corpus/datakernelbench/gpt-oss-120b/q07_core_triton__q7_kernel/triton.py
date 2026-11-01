import triton
import triton.language as tl

TYPES = {
    "supp_ptr": "*i32",
    "cust_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "year_boundary": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    supp_ptr,               # int32 *
    cust_ptr,               # int32 *
    shipdate_ptr,           # int32 *
    price_ptr,              # float64 *
    disc_ptr,               # float64 *
    date_lo,                # int32   (scalar)
    date_hi,                # int32   (scalar)
    year_boundary,          # int32   (scalar, first day of 1996)
    out_ptr,                # float64 * (size 8)
    n_elements,             # int64   (scalar)
    BLOCK_SIZE: tl.constexpr = 8192,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load shipdate and apply date filter
    ship = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    keep = mask & (ship >= date_lo) & (ship <= date_hi)

    # conditional loads for rows that satisfy the filter
    supp = tl.load(supp_ptr + offs, mask=keep, other=0)
    cust = tl.load(cust_ptr + offs, mask=keep, other=0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)

    # compute revenue contribution
    vol = price * (1.0 - disc)

    # determine year: 1995 or 1996
    year = tl.where(ship >= year_boundary, 1996, 1995)
    year_offset = year - 1995          # 0 or 1

    # compact group key : supp*4 + cust*2 + year_offset  (0..7)
    key = supp * 4 + cust * 2 + year_offset

    # scatter‑add into per‑group accumulator
    tl.atomic_add(out_ptr + key, vol)
