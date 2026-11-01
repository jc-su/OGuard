import triton
import triton.language as tl

TYPES = {
    "suppkey_ptr": "*i64",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "suppkey_to_pos_ptr": "*i32",
    "rev_ptr": "*fp64",
    "present_ptr": "*i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "n_suppliers": "i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 8


@triton.jit
def kernel(
    suppkey_ptr, price_ptr, disc_ptr, shipdate_ptr,
    suppkey_to_pos_ptr, rev_ptr, present_ptr,
    date_lo, date_hi, n_suppliers, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate and filter by date range
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    keep_date = (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Load suppkey
    suppkey = tl.load(suppkey_ptr + offs, mask=mask, other=0)
    
    # Lookup position in sorted supplier array using pre-computed mapping
    pos = tl.load(suppkey_to_pos_ptr + suppkey, mask=mask, other=-1)
    valid_supplier = (pos >= 0) & (pos < n_suppliers)
    
    # Combined mask for valid rows
    keep = keep_date & valid_supplier & mask
    
    # Load price and discount only for kept rows
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    
    revenue = price * (1.0 - disc)
    
    # Atomic add to the correct supplier bucket
    tl.atomic_add(rev_ptr + pos, revenue, mask=keep)
    tl.atomic_or(present_ptr + pos, 1, mask=keep)
