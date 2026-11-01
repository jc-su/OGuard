import triton
import triton.language as tl

TYPES = {
    "shipmode_ptr": "*i32",
    "priority_ptr": "*i32",
    "commit_ptr": "*i32",
    "receipt_ptr": "*i32",
    "ship_ptr": "*i32",
    "receipt_lo": "i32",
    "receipt_hi": "i32",
    "high_counts_ptr": "*i64",
    "low_counts_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    shipmode_ptr, priority_ptr, commit_ptr, receipt_ptr, ship_ptr,
    receipt_lo, receipt_hi,
    high_counts_ptr, low_counts_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Evaluate receiptdate first (selectivity ~15% for the year filter)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=0)
    keep = mask & (receipt >= receipt_lo) & (receipt < receipt_hi)

    # Conditionally load commitdate
    commit = tl.load(commit_ptr + offs, mask=keep, other=0)
    keep = keep & (commit < receipt)

    # Conditionally load shipdate
    ship = tl.load(ship_ptr + offs, mask=keep, other=0)
    keep = keep & (ship < commit)

    # Conditionally load shipmode and priority (encoded as int32)
    shipmode = tl.load(shipmode_ptr + offs, mask=keep, other=-1)
    priority = tl.load(priority_ptr + offs, mask=keep, other=-1)

    is_mail = (shipmode == 0)
    is_ship = (shipmode == 1)
    is_high = (priority == 1)
    is_low  = (priority == 0)

    # Compute block-local sums for each category and group
    c_h0 = tl.sum(tl.where(keep & is_mail & is_high, 1, 0)).to(tl.int64)
    c_l0 = tl.sum(tl.where(keep & is_mail & is_low,  1, 0)).to(tl.int64)
    c_h1 = tl.sum(tl.where(keep & is_ship & is_high, 1, 0)).to(tl.int64)
    c_l1 = tl.sum(tl.where(keep & is_ship & is_low,  1, 0)).to(tl.int64)

    # Global accumulation
    tl.atomic_add(high_counts_ptr + 0, c_h0)
    tl.atomic_add(low_counts_ptr + 0, c_l0)
    tl.atomic_add(high_counts_ptr + 1, c_h1)
    tl.atomic_add(low_counts_ptr + 1, c_l1)
