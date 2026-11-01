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
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 8


@triton.jit
def kernel(
    shipmode_ptr,      # int32 *
    priority_ptr,      # int32 *
    commit_ptr,        # int32 *
    receipt_ptr,       # int32 *
    ship_ptr,          # int32 *
    receipt_lo,        # int32 scalar
    receipt_hi,        # int32 scalar
    out_high_ptr,      # int64 *
    out_low_ptr,       # int64 *
    n_elements,        # int64 scalar
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load date columns (masked)
    commit = tl.load(commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=0)
    ship = tl.load(ship_ptr + offs, mask=mask, other=0)

    # Date predicates
    keep = (
        mask
        & (commit < receipt)
        & (ship < commit)
        & (receipt >= receipt_lo)
        & (receipt < receipt_hi)
    )

    # Load cheap columns only for rows that satisfy date predicates
    shipmode = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    priority = tl.load(priority_ptr + offs, mask=keep, other=0)   # 1 = high, 0 = low

    # Per‑lane contributions (int64)
    high0 = tl.where(keep & (shipmode == 0) & (priority == 1), 1, 0).to(tl.int64)
    high1 = tl.where(keep & (shipmode == 1) & (priority == 1), 1, 0).to(tl.int64)
    low0  = tl.where(keep & (shipmode == 0) & (priority == 0), 1, 0).to(tl.int64)
    low1  = tl.where(keep & (shipmode == 1) & (priority == 0), 1, 0).to(tl.int64)

    # Block‑wide reductions
    s_high0 = tl.sum(high0)
    s_high1 = tl.sum(high1)
    s_low0  = tl.sum(low0)
    s_low1  = tl.sum(low1)

    # One atomic add per block per group
    tl.atomic_add(out_high_ptr + 0, s_high0)
    tl.atomic_add(out_high_ptr + 1, s_high1)
    tl.atomic_add(out_low_ptr + 0,  s_low0)
    tl.atomic_add(out_low_ptr + 1,  s_low1)
