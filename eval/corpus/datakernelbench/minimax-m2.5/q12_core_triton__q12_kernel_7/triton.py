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
    "high_out_ptr": "*i64",
    "low_out_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    shipmode_ptr, priority_ptr,
    commit_ptr, receipt_ptr, ship_ptr,
    receipt_lo, receipt_hi,
    high_out_ptr, low_out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load columns
    shipmode = tl.load(shipmode_ptr + offs, mask=mask, other=0)
    priority = tl.load(priority_ptr + offs, mask=mask, other=0)
    commit = tl.load(commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=0)
    ship = tl.load(ship_ptr + offs, mask=mask, other=0)

    # filter predicate
    keep = (
        mask &
        (commit < receipt) &
        (ship < commit) &
        (receipt >= receipt_lo) &
        (receipt < receipt_hi)
    )

    # cast priority to int64 for accumulation
    priority_i64 = priority.to(tl.int64)

    # high and low bits
    high_bit = tl.where(keep, priority_i64, 0)
    low_bit = tl.where(keep, 1 - priority_i64, 0)

    # contributions per group
    high_for_0 = tl.where(keep & (shipmode == 0), high_bit, 0)
    high_for_1 = tl.where(keep & (shipmode == 1), high_bit, 0)
    low_for_0 = tl.where(keep & (shipmode == 0), low_bit, 0)
    low_for_1 = tl.where(keep & (shipmode == 1), low_bit, 0)

    # sum across threads in block
    high_sum_0 = tl.sum(high_for_0)
    high_sum_1 = tl.sum(high_for_1)
    low_sum_0 = tl.sum(low_for_0)
    low_sum_1 = tl.sum(low_for_1)

    # atomic add to output
    tl.atomic_add(high_out_ptr, high_sum_0)
    tl.atomic_add(high_out_ptr + 1, high_sum_1)
    tl.atomic_add(low_out_ptr, low_sum_0)
    tl.atomic_add(low_out_ptr + 1, low_sum_1)
