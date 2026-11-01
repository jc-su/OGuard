import triton
import triton.language as tl

TYPES = {
    "supp_ptr": "*i32",
    "cust_ptr": "*i32",
    "ship_ptr": "*i32",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "lo_day": "i32",
    "hi_day": "i32",
    "day_1996": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 4


@triton.jit
def kernel(
    supp_ptr,                # int32 *
    cust_ptr,                # int32 *
    ship_ptr,                # int32 *
    price_ptr,               # float64 *
    disc_ptr,                # float64 *
    lo_day,                  # i32   scalar
    hi_day,                  # i32   scalar
    day_1996,                # i32   scalar (first day of 1996)
    out_ptr,                 # float64 *
    n_elements,              # i64   total rows
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns
    supp = tl.load(supp_ptr + offs, mask=mask, other=0).to(tl.int32)
    cust = tl.load(cust_ptr + offs, mask=mask, other=0).to(tl.int32)
    ship = tl.load(ship_ptr + offs, mask=mask, other=0).to(tl.int32)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)

    # Predicate on shipdate range
    keep = mask & (ship >= lo_day) & (ship <= hi_day)

    # Year offset: 0 for 1995, 1 for 1996
    year_off = tl.where(ship >= day_1996, 1, 0)

    # Composite key in [0,7]
    key = (supp * 4 + cust * 2 + year_off).to(tl.int64)

    # Compute volume
    volume = price * (1.0 - disc)

    # Add contribution (zero for filtered rows)
    tl.atomic_add(out_ptr + key, tl.where(keep, volume, 0.0))
