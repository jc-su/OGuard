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
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 4


@triton.jit
def kernel(
    supp_ptr, cust_ptr, shipdate_ptr, price_ptr, discount_ptr,
    date_lo, date_hi, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr = 4096,
):
    """
    Fused Q7 kernel: filter + year extraction + volume calc + group aggregation.
    
    Optimizations:
    - Simplified year extraction (1995/1996 only based on day threshold)
    - Direct group index computation (no loop)
    - Single atomic_add per element
    - Fixed block size for H100 (no autotune overhead)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load all columns (coalesced access)
    supp = tl.load(supp_ptr + offs, mask=mask, other=0)
    cust = tl.load(cust_ptr + offs, mask=mask, other=0)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi + 1)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    
    # Filter: shipdate range
    date_ok = (shipdate >= date_lo) & (shipdate <= date_hi)
    
    # Filter: cross-nation pairs only (supp != cust, both in {0,1})
    nation_ok = (supp != cust) & (supp <= 1) & (cust <= 1)
    
    keep = date_ok & nation_ok
    
    # Simplified year extraction for 1995-1996 range
    # 1996-01-01 = 9496 days since Unix epoch
    year_offset = tl.where(shipdate < 9496, 0, 1).to(tl.int64)
    
    # Direct group index: supp*4 + cust*2 + year_offset (0-7)
    group = (supp.to(tl.int64) * 4 + cust.to(tl.int64) * 2 + year_offset).to(tl.int64)
    
    # Compute volume = price * (1 - discount)
    volume = price * (1.0 - discount)
    
    # Single atomic_add per thread to computed group
    vol_contrib = tl.where(keep, volume, 0.0)
    tl.atomic_add(out_ptr + group, vol_contrib)
