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
    "high_ptr": "*i64",
    "low_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    shipmode_ptr,
    priority_ptr,
    commit_ptr,
    receipt_ptr,
    ship_ptr,
    receipt_lo,
    receipt_hi,
    high_ptr,
    low_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns
    shipmode = tl.load(shipmode_ptr + offs, mask=mask, other=0)
    priority = tl.load(priority_ptr + offs, mask=mask, other=0)
    commit = tl.load(commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=0)
    ship = tl.load(ship_ptr + offs, mask=mask, other=0)

    # Filter predicate
    keep = mask & (commit < receipt) & (ship < commit) & (receipt >= receipt_lo) & (receipt < receipt_hi)

    # Cast to int64 for accumulation
    priority_i64 = priority.to(tl.int64)

    # Contributions
    high_contrib = tl.where(keep, priority_i64, 0)
    low_contrib = tl.where(keep, 1 - priority_i64, 0)

    # Group 0 (MAIL)
    high_g0 = tl.sum(tl.where(shipmode == 0, high_contrib, 0))
    low_g0 = tl.sum(tl.where(shipmode == 0, low_contrib, 0))
    # Group 1 (SHIP)
    high_g1 = tl.sum(tl.where(shipmode == 1, high_contrib, 0))
    low_g1 = tl.sum(tl.where(shipmode == 1, low_contrib, 0))

    # Atomic add to global counters
    tl.atomic_add(high_ptr + 0, high_g0)
    tl.atomic_add(low_ptr + 0, low_g0)
    tl.atomic_add(high_ptr + 1, high_g1)
    tl.atomic_add(low_ptr + 1, low_g1)
