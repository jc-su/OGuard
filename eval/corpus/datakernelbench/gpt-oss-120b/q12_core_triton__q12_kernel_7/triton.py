import triton
import triton.language as tl

TYPES = {
    "shipmode_ptr": "*i32",
    "high_flag_ptr": "*i32",
    "commit_ptr": "*i32",
    "receipt_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "receipt_lo": "i32",
    "receipt_hi": "i32",
    "out_high_ptr": "*i64",
    "out_low_ptr": "*i64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    shipmode_ptr,            # int32 *
    high_flag_ptr,           # int32 *
    commit_ptr,              # int32 *
    receipt_ptr,             # int32 *
    shipdate_ptr,            # int32 *
    receipt_lo: tl.int32,    # scalar bound
    receipt_hi: tl.int32,    # scalar bound
    out_high_ptr,            # int64 *
    out_low_ptr,             # int64 *
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load the three date columns (masked)
    commit   = tl.load(commit_ptr   + offs, mask=mask,    other=0)
    receipt  = tl.load(receipt_ptr  + offs, mask=mask,    other=0)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask,    other=0)

    # Predicate evaluation (int32)
    pred = (commit < receipt) & (shipdate < commit) & \
           (receipt >= receipt_lo) & (receipt < receipt_hi)

    keep = mask & pred

    # Conditional loads for shipmode and high flag (masked)
    shipmode = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    high_flag = tl.load(high_flag_ptr + offs, mask=keep, other=0)

    # Convert flags to int64 while respecting the mask
    high_i = tl.where(keep, tl.cast(high_flag, tl.int64), 0)
    low_i  = tl.where(keep, tl.cast(1 - high_flag, tl.int64), 0)

    # Atomic adds per shipmode (0 = MAIL, 1 = SHIP)
    tl.atomic_add(out_high_ptr + shipmode, high_i)
    tl.atomic_add(out_low_ptr  + shipmode, low_i)
