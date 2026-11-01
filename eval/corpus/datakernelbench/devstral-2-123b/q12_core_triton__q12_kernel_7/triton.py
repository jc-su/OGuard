import triton
import triton.language as tl

TYPES = {
    "l_commitdate_ptr": "*i32",
    "l_receiptdate_ptr": "*i32",
    "l_shipdate_ptr": "*i32",
    "shipmode_ptr": "*i32",
    "o_orderpriority_ptr": "*i32",
    "receipt_lo": "i32",
    "receipt_hi": "i32",
    "n_elements": "i32",
    "high_out_ptr": "*i64",
    "low_out_ptr": "*i64",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_commitdate_ptr, l_receiptdate_ptr, l_shipdate_ptr, shipmode_ptr,
    o_orderpriority_ptr, receipt_lo, receipt_hi, n_elements,
    high_out_ptr, low_out_ptr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load date columns
    commit = tl.load(l_commitdate_ptr + offs, mask=mask, other=receipt_hi)
    receipt = tl.load(l_receiptdate_ptr + offs, mask=mask, other=receipt_hi)
    ship = tl.load(l_shipdate_ptr + offs, mask=mask, other=receipt_hi)

    # Filter condition
    keep = (
        mask
        & (commit < receipt)
        & (ship < commit)
        & (receipt >= receipt_lo)
        & (receipt < receipt_hi)
    )

    # Load shipmode and priority for kept rows
    shipmode = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    priority = tl.load(o_orderpriority_ptr + offs, mask=keep, other=0)

    # Compute group counts using block-level accumulation
    for sm in tl.static_range(2):
        sm_mask = (shipmode == sm) & keep
        high_mask = sm_mask & (priority > 0)

        # Reduce within warp
        high_sum = tl.sum(high_mask, axis=0)
        low_sum = tl.sum(sm_mask, axis=0) - high_sum

        # Store results using atomic add
        tl.atomic_add(high_out_ptr + sm, high_sum)
        tl.atomic_add(low_out_ptr + sm, low_sum)
