import triton
import triton.language as tl

TYPES = {
    "l_partkey_ptr": "*i64",
    "l_suppkey_ptr": "*i64",
    "l_quantity_ptr": "*fp64",
    "l_shipdate_days_ptr": "*i32",
    "counts_ptr": "*i32",
    "supp_slots_ptr": "*i64",
    "idx_slots_ptr": "*i64",
    "sum_qty_ptr": "*fp64",
    "n_lineitem": "i32",
    "n_parts": "i32",
    "shipdate_lo_days": "i32",
    "shipdate_hi_days": "i32",
}

CONSTANTS = {"max_slots": 4, "BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_partkey_ptr,
    l_suppkey_ptr,
    l_quantity_ptr,
    l_shipdate_days_ptr,
    counts_ptr,
    supp_slots_ptr,
    idx_slots_ptr,
    sum_qty_ptr,
    n_lineitem,
    n_parts,
    shipdate_lo_days,
    shipdate_hi_days,
    max_slots: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_lineitem

    ship_days = tl.load(l_shipdate_days_ptr + offs, mask=in_bounds, other=shipdate_hi_days)
    date_keep = in_bounds & (ship_days >= shipdate_lo_days) & (ship_days < shipdate_hi_days)

    p = tl.load(l_partkey_ptr + offs, mask=date_keep, other=0)
    s = tl.load(l_suppkey_ptr + offs, mask=date_keep, other=-1)

    valid_part = date_keep & (p >= 0) & (p < n_parts)
    cnt = tl.load(counts_ptr + p, mask=valid_part, other=0)
    has_filtered_part = valid_part & (cnt > 0)

    base = p * max_slots
    found_idx = tl.full((BLOCK_SIZE,), -1, tl.int64)

    for j in tl.static_range(0, max_slots):
        ps_supp = tl.load(supp_slots_ptr + base + j, mask=has_filtered_part, other=-1)
        ps_idx = tl.load(idx_slots_ptr + base + j, mask=has_filtered_part, other=-1)
        hit = has_filtered_part & (s == ps_supp)
        found_idx = tl.where(hit, ps_idx, found_idx)

    matched = found_idx >= 0
    qty = tl.load(l_quantity_ptr + offs, mask=matched, other=0.0)
    tl.atomic_add(sum_qty_ptr + found_idx, qty, sem="relaxed", mask=matched)
