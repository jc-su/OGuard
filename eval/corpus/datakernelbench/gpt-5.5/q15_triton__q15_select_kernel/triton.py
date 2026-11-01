import triton
import triton.language as tl

TYPES = {
    "totals_ptr": "*fp64",
    "present_ptr": "*u8",
    "sorted_suppkey_ptr": "*i64",
    "max_ptr": "*fp64",
    "count_ptr": "*i32",
    "sel_key_ptr": "*i64",
    "sel_rev_ptr": "*fp64",
}

CONSTANTS = {"n_sup": 100000, "BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    totals_ptr,
    present_ptr,
    sorted_suppkey_ptr,
    max_ptr,
    count_ptr,
    sel_key_ptr,
    sel_rev_ptr,
    n_sup: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_sup

    max_rev = tl.load(max_ptr)
    present = tl.load(present_ptr + offs, mask=mask, other=0)
    vals = tl.load(totals_ptr + offs, mask=mask, other=0.0)

    selected = mask & (present != 0) & (vals == max_rev)
    flags = tl.where(selected, 1, 0)
    n_selected = tl.sum(flags, axis=0)
    prefix = tl.cumsum(flags, axis=0) - 1
    base = tl.atomic_add(count_ptr, n_selected, sem="relaxed")

    keys = tl.load(sorted_suppkey_ptr + offs, mask=selected, other=0)
    out_pos = base + prefix
    tl.store(sel_key_ptr + out_pos, keys, mask=selected)
    tl.store(sel_rev_ptr + out_pos, vals, mask=selected)
