import triton
import triton.language as tl

TYPES = {
    "supp_ptr": "*i32",
    "cust_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "year_boundary": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 4


@triton.jit
def kernel(
    supp_ptr, cust_ptr, shipdate_ptr, price_ptr, discount_ptr,
    date_lo, date_hi, year_boundary,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate first to filter early
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi + 1)
    
    # Filter by shipdate range (inclusive)
    keep = mask & (shipdate >= date_lo) & (shipdate <= date_hi)
    
    # Simplified year extraction for 1995-1996 range
    # year_boundary = day number for 1996-01-01 (9496)
    # If shipdate < year_boundary: year = 1995 (offset 0), else year = 1996 (offset 1)
    year_offset = tl.where(shipdate < year_boundary, 0, 1)
    
    # Only load other columns for rows that pass date filter
    supp = tl.load(supp_ptr + offs, mask=keep, other=0)
    cust = tl.load(cust_ptr + offs, mask=keep, other=0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    
    # Build composite group key: supp*4 + cust*2 + year_offset
    # supp in {0,1}, cust in {0,1}, year_offset in {0,1} => 8 possible groups
    key = supp.to(tl.int64) * 4 + cust.to(tl.int64) * 2 + year_offset.to(tl.int64)
    
    # volume = price * (1 - discount)
    volume = price * (1.0 - discount)
    
    # Single atomic_add per element to the computed key
    tl.atomic_add(out_ptr + key, volume, mask=keep)
