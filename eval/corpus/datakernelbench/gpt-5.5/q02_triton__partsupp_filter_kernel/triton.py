import triton
import triton.language as tl

TYPES = {
    "ps_partkey_ptr": "*i64",
    "ps_suppkey_ptr": "*i64",
    "part_flags_ptr": "*i8",
    "supp_flags_ptr": "*i8",
    "out_mask_ptr": "*i1",
    "n_rows": "i32",
    "n_part_flags": "i32",
    "n_supp_flags": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    ps_partkey_ptr,
    ps_suppkey_ptr,
    part_flags_ptr,
    supp_flags_ptr,
    out_mask_ptr,
    n_rows,
    n_part_flags,
    n_supp_flags,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    partkey = tl.load(ps_partkey_ptr + offs, mask=mask, other=-1)
    suppkey = tl.load(ps_suppkey_ptr + offs, mask=mask, other=-1)

    valid_part = mask & (partkey >= 0) & (partkey < n_part_flags)
    valid_supp = mask & (suppkey >= 0) & (suppkey < n_supp_flags)

    safe_part = tl.where(valid_part, partkey, 0)
    safe_supp = tl.where(valid_supp, suppkey, 0)

    part_ok = tl.load(part_flags_ptr + safe_part, mask=valid_part, other=0)
    supp_ok = tl.load(supp_flags_ptr + safe_supp, mask=valid_supp, other=0)

    keep = mask & (part_ok != 0) & (supp_ok != 0)
    tl.store(out_mask_ptr + offs, keep, mask=mask)
