import triton
import triton.language as tl

TYPES = {
    "l_suppkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_days_ptr": "*i32",
    "totals_ptr": "*fp64",
    "present_ptr": "*i8",
    "n_elements": "i32",
    "n_suppliers": "i32",
    "shipdate_lo_days": "i32",
    "shipdate_hi_days": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_suppkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_days_ptr,
    totals_ptr,
    present_ptr,
    n_elements,
    n_suppliers,
    shipdate_lo_days,
    shipdate_hi_days,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ship_days = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_hi_days)
    keep = mask & (ship_days >= shipdate_lo_days) & (ship_days < shipdate_hi_days)

    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)
    idx = suppkey - 1
    valid = keep & (idx >= 0) & (idx < n_suppliers)

    price = tl.load(l_extendedprice_ptr + offs, mask=valid, other=0.0)
    disc = tl.load(l_discount_ptr + offs, mask=valid, other=0.0)
    revenue = price * (1.0 - disc)

    tl.atomic_add(totals_ptr + idx, revenue, sem="relaxed", mask=valid)
    tl.store(present_ptr + idx, 1, mask=valid)
