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
    "out_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 8


@triton.jit
def kernel(
    shipmode_ptr, priority_ptr,
    commit_ptr, receipt_ptr, ship_ptr,
    receipt_lo, receipt_hi,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load date columns (int32)
    commit = tl.load(commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=0)
    ship = tl.load(ship_ptr + offs, mask=mask, other=0)

    # Date predicates
    # l_commitdate < l_receiptdate
    p1 = commit < receipt
    # l_shipdate < l_commitdate
    p2 = ship < commit
    # l_receiptdate >= receipt_lo
    p3 = receipt >= receipt_lo
    # l_receiptdate < receipt_hi
    p4 = receipt < receipt_hi

    keep = p1 & p2 & p3 & p4

    # Load shipmode and priority (int32)
    # 0=MAIL, 1=SHIP for shipmode
    # 1=High, 0=Low for priority
    shipmode = tl.load(shipmode_ptr + offs, mask=mask, other=0)
    priority = tl.load(priority_ptr + offs, mask=mask, other=0)

    # Compute output index
    # Output layout: [high_mail, high_ship, low_mail, low_ship]
    # idx = shipmode + (1 - priority) * 2
    # If priority=1 (High): idx = shipmode (0 or 1)
    # If priority=0 (Low): idx = shipmode + 2 (2 or 3)
    idx = shipmode + (1 - priority) * 2

    # Per-block accumulation for 4 buckets using scalars
    high_mail = tl.sum(tl.where((idx == 0) & keep, 1, 0).to(tl.int32))
    high_ship = tl.sum(tl.where((idx == 1) & keep, 1, 0).to(tl.int32))
    low_mail = tl.sum(tl.where((idx == 2) & keep, 1, 0).to(tl.int32))
    low_ship = tl.sum(tl.where((idx == 3) & keep, 1, 0).to(tl.int32))

    # Atomic add block counts to global output
    tl.atomic_add(out_ptr + 0, high_mail)
    tl.atomic_add(out_ptr + 1, high_ship)
    tl.atomic_add(out_ptr + 2, low_mail)
    tl.atomic_add(out_ptr + 3, low_ship)
