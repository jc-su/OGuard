import triton
import triton.language as tl

TYPES = {
    "shipmode_code_ptr": "*i32",
    "priority_high_ptr": "*i32",
    "commitdate_ptr": "*i32",
    "receiptdate_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "receipt_lo_days": "i32",
    "receipt_hi_days": "i32",
    "high_out_ptr": "*i64",
    "low_out_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    shipmode_code_ptr,
    priority_high_ptr,
    commitdate_ptr,
    receiptdate_ptr,
    shipdate_ptr,
    receipt_lo_days,
    receipt_hi_days,
    high_out_ptr,
    low_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offs < n_elements

    receipt = tl.load(receiptdate_ptr + offs, mask=active, other=receipt_lo_days - 1)
    commit = tl.load(commitdate_ptr + offs, mask=active, other=0)
    ship = tl.load(shipdate_ptr + offs, mask=active, other=0)

    keep = (
        active
        & (commit < receipt)
        & (ship < commit)
        & (receipt >= receipt_lo_days)
        & (receipt < receipt_hi_days)
    )

    code = tl.load(shipmode_code_ptr + offs, mask=keep, other=-1)
    high = tl.load(priority_high_ptr + offs, mask=keep, other=0)

    is_mail = keep & (code == 0)
    is_ship = keep & (code == 1)

    high_mail = tl.sum(tl.where(is_mail, high, 0), axis=0)
    high_ship = tl.sum(tl.where(is_ship, high, 0), axis=0)
    low_mail = tl.sum(tl.where(is_mail, 1 - high, 0), axis=0)
    low_ship = tl.sum(tl.where(is_ship, 1 - high, 0), axis=0)

    tl.atomic_add(high_out_ptr + 0, high_mail.to(tl.int64), sem="relaxed")
    tl.atomic_add(high_out_ptr + 1, high_ship.to(tl.int64), sem="relaxed")
    tl.atomic_add(low_out_ptr + 0, low_mail.to(tl.int64), sem="relaxed")
    tl.atomic_add(low_out_ptr + 1, low_ship.to(tl.int64), sem="relaxed")
