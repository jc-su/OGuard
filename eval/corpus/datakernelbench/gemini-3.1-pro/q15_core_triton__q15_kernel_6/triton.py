import triton
import triton.language as tl

TYPES = {
    "l_suppkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_ptr": "*i32",
    "rev_by_suppkey_ptr": "*fp64",
    "counts_by_suppkey_ptr": "*i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "max_s": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_suppkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_ptr,
    rev_by_suppkey_ptr,
    counts_by_suppkey_ptr,
    date_lo,
    date_hi,
    max_s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Filter early on shipdate
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Load suppkey only for rows that passed date filter
    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)
    # Ensure suppkey is within the allocated domain bounds
    valid = keep & (suppkey >= 0) & (suppkey <= max_s)
    
    # Load revenue components for valid rows
    price = tl.load(l_extendedprice_ptr + offs, mask=valid, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=valid, other=1.0)
    
    # Compute revenue
    rev = tl.where(valid, price * (1.0 - discount), 0.0)
    
    # Atomic add to the global suppkey buckets
    tl.atomic_add(rev_by_suppkey_ptr + suppkey, rev, mask=valid)
    tl.atomic_add(counts_by_suppkey_ptr + suppkey, 1, mask=valid)
