import triton
import triton.language as tl

TYPES = {
    "s_suppkey_ptr": "*i64",
    "s_row_idx_ptr": "*i64",
    "supp_set_ptr": "*i64",
    "out_idx_ptr": "*i64",
    "count_ptr": "*i64",
    "n_supplier": "i32",
    "table_mask": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256, "MAX_PROBE": 64}

NUM_WARPS = 4


@triton.jit
def kernel(
    s_suppkey_ptr,
    s_row_idx_ptr,
    supp_set_ptr,
    out_idx_ptr,
    count_ptr,
    n_supplier,
    table_mask,
    BLOCK_SIZE: tl.constexpr,
    MAX_PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_supplier

    supp = tl.load(s_suppkey_ptr + offs, mask=mask, other=-1)

    x = supp * 1009837
    x = x ^ (x >> 16)
    h = x & table_mask

    found = tl.full((BLOCK_SIZE,), False, tl.int1)
    miss = tl.full((BLOCK_SIZE,), False, tl.int1)

    for probe in tl.static_range(0, MAX_PROBE):
        active = mask & (~found) & (~miss)
        slot = (h + probe) & table_mask
        cand = tl.load(supp_set_ptr + slot, mask=active, other=-1)
        match = active & (cand == supp)
        empty = active & (cand == -1)
        found = found | match
        miss = miss | empty

    found_i64 = found.to(tl.int64)
    prefix = tl.cumsum(found_i64, 0) - 1
    block_total = tl.sum(found_i64, 0)
    base = tl.atomic_add(count_ptr, block_total, sem="relaxed")
    row_idx = tl.load(s_row_idx_ptr + offs, mask=found, other=0)
    tl.store(out_idx_ptr + base + prefix, row_idx, mask=found)
