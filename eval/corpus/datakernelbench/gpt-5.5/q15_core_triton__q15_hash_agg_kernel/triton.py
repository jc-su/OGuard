import triton
import triton.language as tl

TYPES = {
    "l_suppkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_days_ptr": "*i32",
    "hash_keys_ptr": "*i64",
    "hash_vals_ptr": "*i64",
    "totals_ptr": "*fp64",
    "present_ptr": "*i8",
    "n_elements": "i32",
    "table_mask": "i32",
    "shipdate_lo_days": "i32",
    "shipdate_hi_days": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024, "MAX_PROBES": 128}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_suppkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_days_ptr,
    hash_keys_ptr,
    hash_vals_ptr,
    totals_ptr,
    present_ptr,
    n_elements,
    table_mask,
    shipdate_lo_days,
    shipdate_hi_days,
    BLOCK_SIZE: tl.constexpr,
    MAX_PROBES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ship_days = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_hi_days)
    keep = mask & (ship_days >= shipdate_lo_days) & (ship_days < shipdate_hi_days)

    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)
    h = (suppkey * 1103515245) & table_mask

    found_idx = tl.full((BLOCK_SIZE,), -1, tl.int64)
    active = keep

    for _ in tl.static_range(0, MAX_PROBES):
        slot_key = tl.load(hash_keys_ptr + h, mask=active, other=-1)
        hit = active & (slot_key == suppkey)
        empty = active & (slot_key == -1)

        slot_val = tl.load(hash_vals_ptr + h, mask=hit, other=-1)
        found_idx = tl.where(hit, slot_val, found_idx)

        active = active & ~(hit | empty)
        h = (h + 1) & table_mask

    valid = keep & (found_idx >= 0)

    price = tl.load(l_extendedprice_ptr + offs, mask=valid, other=0.0)
    disc = tl.load(l_discount_ptr + offs, mask=valid, other=0.0)
    revenue = price * (1.0 - disc)

    tl.atomic_add(totals_ptr + found_idx, revenue, sem="relaxed", mask=valid)
    tl.store(present_ptr + found_idx, 1, mask=valid)
