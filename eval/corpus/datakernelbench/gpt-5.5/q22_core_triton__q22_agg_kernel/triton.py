import triton
import triton.language as tl

TYPES = {
    "c_custkey_ptr": "*i64",
    "acctbal_ptr": "*fp64",
    "cntrycode_ptr": "*i32",
    "in_mask_ptr": "*i1",
    "bitmap_ptr": "*u8",
    "stats_ptr": "*fp64",
    "counts_ptr": "*i64",
    "sums_ptr": "*fp64",
    "n_elements": "i32",
    "max_key": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    c_custkey_ptr,
    acctbal_ptr,
    cntrycode_ptr,
    in_mask_ptr,
    bitmap_ptr,
    stats_ptr,
    counts_ptr,
    sums_ptr,
    n_elements,
    max_key,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    sum_pos = tl.load(stats_ptr + 0)
    cnt_pos = tl.load(stats_ptr + 1)
    avg_bal = sum_pos / cnt_pos

    key = tl.load(c_custkey_ptr + offs, mask=mask, other=-1)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    code = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_set = tl.load(in_mask_ptr + offs, mask=mask, other=0) != 0

    valid_key = mask & (key >= 0) & (key <= max_key)
    safe_key = tl.where(valid_key, key, 0)
    has_order = tl.load(bitmap_ptr + safe_key, mask=valid_key, other=0) != 0

    valid_code = (code >= 0) & (code < 100)
    keep = mask & in_set & valid_code & (bal > avg_bal) & (~has_order)

    one = tl.full((BLOCK_SIZE,), 1, tl.int64)
    tl.atomic_add(counts_ptr + code, one, sem="relaxed", mask=keep)
    tl.atomic_add(sums_ptr + code, bal, sem="relaxed", mask=keep)
