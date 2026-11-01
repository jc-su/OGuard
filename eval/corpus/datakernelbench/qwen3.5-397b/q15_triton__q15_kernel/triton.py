import triton
import triton.language as tl

TYPES = {
    "shipdate_ptr": "*i32",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "supp_ptr": "*i64",
    "totals_ptr": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 4


@triton.jit
def kernel(
    shipdate_ptr,
    price_ptr,
    disc_ptr,
    supp_ptr,
    totals_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 8192,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    supp = tl.load(supp_ptr + offs, mask=mask, other=0)
    
    date_ok = (shipdate >= date_lo) & (shipdate < date_hi)
    valid = date_ok & (supp > 0)
    
    revenue = price * (1.0 - disc)
    rev_val = tl.where(valid, revenue, 0.0)
    
    tl.atomic_add(totals_ptr + supp, rev_val)
