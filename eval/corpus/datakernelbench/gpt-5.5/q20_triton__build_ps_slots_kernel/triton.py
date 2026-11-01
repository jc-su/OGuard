import triton
import triton.language as tl

TYPES = {
    "ps_partkey_ptr": "*i64",
    "ps_suppkey_ptr": "*i64",
    "counts_ptr": "*i32",
    "supp_slots_ptr": "*i64",
    "idx_slots_ptr": "*i64",
}

CONSTANTS = {"max_slots": 4}

NUM_WARPS = 1


@triton.jit
def kernel(
    ps_partkey_ptr,
    ps_suppkey_ptr,
    counts_ptr,
    supp_slots_ptr,
    idx_slots_ptr,
    max_slots: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.load(ps_partkey_ptr + row)
    s = tl.load(ps_suppkey_ptr + row)

    slot = tl.atomic_add(counts_ptr + p, 1, sem="relaxed")
    ok = slot < max_slots
    out_pos = p * max_slots + slot

    tl.store(supp_slots_ptr + out_pos, s, mask=ok)
    tl.store(idx_slots_ptr + out_pos, row, mask=ok)
