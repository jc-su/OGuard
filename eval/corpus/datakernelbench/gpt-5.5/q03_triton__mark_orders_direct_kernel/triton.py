import triton
import triton.language as tl

TYPES = {
    "o_orderkey_ptr": "*i64",
    "o_orderdate_days_ptr": "*i32",
    "o_shippriority_ptr": "*i32",
    "order_flag_ptr": "*u8",
    "orderdate_table_ptr": "*i32",
    "shippriority_table_ptr": "*i32",
    "n_orders": "i32",
    "capacity": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_orderkey_ptr,
    o_orderdate_days_ptr,
    o_shippriority_ptr,
    order_flag_ptr,
    orderdate_table_ptr,
    shippriority_table_ptr,
    n_orders,
    capacity,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders

    key = tl.load(o_orderkey_ptr + offs, mask=mask, other=-1)
    ok = mask & (key >= 0) & (key < capacity)

    od = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=0)
    sp = tl.load(o_shippriority_ptr + offs, mask=mask, other=0)

    tl.store(order_flag_ptr + key, 1, mask=ok)
    tl.store(orderdate_table_ptr + key, od, mask=ok)
    tl.store(shippriority_table_ptr + key, sp, mask=ok)
