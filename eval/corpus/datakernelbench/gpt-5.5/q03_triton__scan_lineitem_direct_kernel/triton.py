import triton
import triton.language as tl

TYPES = {
    "l_orderkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_days_ptr": "*i32",
    "order_flag_ptr": "*u8",
    "revenue_table_ptr": "*fp64",
    "n_lineitems": "i32",
    "capacity": "i32",
    "shipdate_lo_days": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_orderkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_days_ptr,
    order_flag_ptr,
    revenue_table_ptr,
    n_lineitems,
    capacity,
    shipdate_lo_days,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_lineitems

    shipdate = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_lo_days)
    ship_ok = mask & (shipdate > shipdate_lo_days)

    key = tl.load(l_orderkey_ptr + offs, mask=ship_ok, other=0)
    in_range = ship_ok & (key >= 0) & (key < capacity)

    is_eligible_order = tl.load(order_flag_ptr + key, mask=in_range, other=0)
    keep = in_range & (is_eligible_order != 0)

    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    revenue = price * (1.0 - discount)

    tl.atomic_add(revenue_table_ptr + key, revenue, mask=keep, sem="relaxed")
