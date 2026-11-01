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
    "out_high_ptr": "*i64",
    "out_low_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 8192,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    shipmode_ptr,
    priority_ptr,
    commit_ptr,
    receipt_ptr,
    ship_ptr,
    receipt_lo,
    receipt_hi,
    out_high_ptr,
    out_low_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    commit = tl.load(commit_ptr + offs, mask=mask, other=receipt_hi)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=receipt_lo - 1)
    ship = tl.load(ship_ptr + offs, mask=mask, other=receipt_hi)

    keep = (
        mask
        & (commit < receipt)
        & (ship < commit)
        & (receipt >= receipt_lo)
        & (receipt < receipt_hi)
    )

    code = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    priority = tl.load(priority_ptr + offs, mask=keep, other=0)

    # Sum high and low counts per group
    # Use two 2-element accumulators for high/low counts
    high_group0 = tl.sum(tl.where((code == 0) & keep, priority, 0))
    high_group1 = tl.sum(tl.where((code == 1) & keep, priority, 0))
    low_group0 = tl.sum(tl.where((code == 0) & keep, 1 - priority, 0))
    low_group1 = tl.sum(tl.where((code == 1) & keep, 1 - priority, 0))

    # Atomic add to global accumulators
    tl.atomic_add(out_high_ptr, high_group0)
    tl.atomic_add(out_high_ptr + 1, high_group1)
    tl.atomic_add(out_low_ptr, low_group0)
    tl.atomic_add(out_low_ptr + 1, low_group1)
