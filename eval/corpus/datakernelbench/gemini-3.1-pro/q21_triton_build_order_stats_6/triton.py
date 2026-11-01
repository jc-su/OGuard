import triton
import triton.language as tl

TYPES = {
    "l_orderkey": "*i32",
    "l_suppkey": "*i32",
    "l_receiptdate": "*i32",
    "l_commitdate": "*i32",
    "order_status_F": "*u8",
    "o_supp_min": "*i32",
    "o_supp_max": "*i32",
    "o_late_min": "*i32",
    "o_late_max": "*i32",
    "num_lineitems": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_orderkey, l_suppkey, l_receiptdate, l_commitdate,
    order_status_F,
    o_supp_min, o_supp_max, o_late_min, o_late_max,
    num_lineitems,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_lineitems

    o_key = tl.load(l_orderkey + offs, mask=mask, other=0)
    s_key = tl.load(l_suppkey + offs, mask=mask, other=0)
    receipt = tl.load(l_receiptdate + offs, mask=mask, other=0)
    commit = tl.load(l_commitdate + offs, mask=mask, other=0)

    # Only process lineitems for orders with status 'F'
    is_F = tl.load(order_status_F + o_key, mask=mask, other=0)
    active_mask = mask & (is_F == 1)

    late = receipt > commit

    # All active suppliers per order: atomic min/max
    # Idempotent operation, safe with triton.autotune
    tl.atomic_min(o_supp_min + o_key, s_key, mask=active_mask)
    tl.atomic_max(o_supp_max + o_key, s_key, mask=active_mask)

    # Late active suppliers per order: atomic min/max
    late_mask = active_mask & late
    tl.atomic_min(o_late_min + o_key, s_key, mask=late_mask)
    tl.atomic_max(o_late_max + o_key, s_key, mask=late_mask)
