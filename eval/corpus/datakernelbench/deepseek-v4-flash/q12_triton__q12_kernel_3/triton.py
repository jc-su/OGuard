import triton
import triton.language as tl

TYPES = {
    "shipmode_ptr": "*i32",
    "priority_ptr": "*i32",
    "commitdate_ptr": "*i32",
    "receiptdate_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "receipt_lo": "i32",
    "receipt_hi": "i32",
    "out_high_ptr": "*i64",
    "out_low_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 2048,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    shipmode_ptr, priority_ptr, commitdate_ptr, receiptdate_ptr, shipdate_ptr,
    receipt_lo, receipt_hi, out_high_ptr, out_low_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load filter columns
    commit = tl.load(commitdate_ptr + offs, mask=mask, other=receipt_hi)
    receipt = tl.load(receiptdate_ptr + offs, mask=mask, other=receipt_lo)
    ship = tl.load(shipdate_ptr + offs, mask=mask, other=receipt_hi)
    
    # Predicates
    keep = mask
    keep = keep & (commit < receipt)
    keep = keep & (ship < commit)
    keep = keep & (receipt >= receipt_lo)
    keep = keep & (receipt < receipt_hi)
    
    # Load projection columns — only for rows that pass filter
    mode = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    prio = tl.load(priority_ptr + offs, mask=keep, other=0)
    
    # Compute per-group sums
    # For group 0 (MAIL, shipmode=0): sum of high_flag and low_flag
    high0_sum = tl.sum(tl.where(keep & (mode == 0), prio.to(tl.int64), 0))
    low0_sum = tl.sum(tl.where(keep & (mode == 0), (1 - prio.to(tl.int64)), 0))
    
    # For group 1 (SHIP, shipmode=1): sum of high_flag and low_flag  
    high1_sum = tl.sum(tl.where(keep & (mode == 1), prio.to(tl.int64), 0))
    low1_sum = tl.sum(tl.where(keep & (mode == 1), (1 - prio.to(tl.int64)), 0))
    
    # Atomic add to global outputs
    tl.atomic_add(out_high_ptr, high0_sum)
    tl.atomic_add(out_high_ptr + 1, high1_sum)
    tl.atomic_add(out_low_ptr, low0_sum)
    tl.atomic_add(out_low_ptr + 1, low1_sum)
