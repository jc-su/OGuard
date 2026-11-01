import triton
import triton.language as tl

TYPES = {
    "order_key_ptr": "*i64",
    "order_date_ptr": "*i32",
    "priority_ptr": "*i32",
    "out_key_ptr": "*i64",
    "out_priority_ptr": "*i32",
    "out_valid_ptr": "*i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 1024
}

NUM_WARPS = 4


@triton.jit
def kernel(
    order_key_ptr,
    order_date_ptr,
    priority_ptr,
    out_key_ptr,
    out_priority_ptr,
    out_valid_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Filter orders by date range using Triton.
    Fuses date filter with key/priority extraction in one kernel pass.
    Memory-bound optimization: single HBM pass for filter+project.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    order_date = tl.load(order_date_ptr + offs, mask=mask, other=date_hi)
    date_ok = (order_date >= date_lo) & (order_date < date_hi)
    keep = mask & date_ok
    
    order_key = tl.load(order_key_ptr + offs, mask=keep, other=0)
    priority = tl.load(priority_ptr + offs, mask=keep, other=0)
    
    tl.store(out_key_ptr + offs, order_key, mask=keep)
    tl.store(out_priority_ptr + offs, priority, mask=keep)
    tl.store(out_valid_ptr + offs, keep.to(tl.int32), mask=mask)
