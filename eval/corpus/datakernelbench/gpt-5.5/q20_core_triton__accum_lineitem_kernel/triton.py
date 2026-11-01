import triton
import triton.language as tl

TYPES = {
    "l_partkey_ptr": "*i64",
    "l_suppkey_ptr": "*i64",
    "l_quantity_ptr": "*fp64",
    "ht_part_ptr": "*i64",
    "ht_supp_ptr": "*i64",
    "ht_idx_ptr": "*i64",
    "sum_qty_ptr": "*fp64",
    "n_lineitem": "i32",
    "table_mask": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256, "MAX_PROBE": 64}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_partkey_ptr,
    l_suppkey_ptr,
    l_quantity_ptr,
    ht_part_ptr,
    ht_supp_ptr,
    ht_idx_ptr,
    sum_qty_ptr,
    n_lineitem,
    table_mask,
    BLOCK_SIZE: tl.constexpr,
    MAX_PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_lineitem

    p = tl.load(l_partkey_ptr + offs, mask=mask, other=-1)
    s = tl.load(l_suppkey_ptr + offs, mask=mask, other=-1)
    q = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)

    x = (p * 1000003) ^ (s * 1009837)
    x = x ^ (x >> 16)
    h = x & table_mask

    found = tl.full((BLOCK_SIZE,), False, tl.int1)
    miss = tl.full((BLOCK_SIZE,), False, tl.int1)
    found_idx = tl.full((BLOCK_SIZE,), 0, tl.int64)

    for probe in tl.static_range(0, MAX_PROBE):
        active = mask & (~found) & (~miss)
        slot = (h + probe) & table_mask
        cand = tl.load(ht_idx_ptr + slot, mask=active, other=-1)
        has_value = cand >= 0
        hp = tl.load(ht_part_ptr + slot, mask=active & has_value, other=-1)
        hs = tl.load(ht_supp_ptr + slot, mask=active & has_value, other=-1)

        match = active & has_value & (hp == p) & (hs == s)
        empty = active & (cand == -1)

        found_idx = tl.where(match, cand, found_idx)
        found = found | match
        miss = miss | empty

    tl.atomic_add(sum_qty_ptr + found_idx, q, mask=found, sem="relaxed")
