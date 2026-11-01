import triton
import triton.language as tl

TYPES = {
    "l_orderkey_ptr": "*i64",
    "l_commitdate_days_ptr": "*i32",
    "l_receiptdate_days_ptr": "*i32",
    "flags_ptr": "*u8",
    "n_elements": "i32",
    "flag_n": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_orderkey_ptr,
    l_commitdate_days_ptr,
    l_receiptdate_days_ptr,
    flags_ptr,
    n_elements,
    flag_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    commit_days = tl.load(l_commitdate_days_ptr + offs, mask=mask, other=0)
    receipt_days = tl.load(l_receiptdate_days_ptr + offs, mask=mask, other=0)
    late = mask & (commit_days < receipt_days)

    keys = tl.load(l_orderkey_ptr + offs, mask=late, other=-1)
    in_range = late & (keys >= 0) & (keys < flag_n)
    one = tl.full((BLOCK_SIZE,), 1, dtype=tl.uint8)
    tl.store(flags_ptr + keys, one, mask=in_range)
