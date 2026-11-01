import triton
import triton.language as tl

TYPES = {
    "l_orderkey_ptr": "*i64",
    "l_commit_ptr": "*i32",
    "l_receipt_ptr": "*i32",
    "bitmap_ptr": "*i32",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    l_orderkey_ptr,          # int64 *
    l_commit_ptr,            # int32 *
    l_receipt_ptr,           # int32 *
    bitmap_ptr,              # int32 *
    n_elements,              # total rows in lineitem
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    commit = tl.load(l_commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(l_receipt_ptr + offs, mask=mask, other=0)

    keep = mask & (commit < receipt)

    orderkey = tl.load(l_orderkey_ptr + offs, mask=keep, other=0)

    # write 1 (as int32) where predicate holds; atomic_max works as set‑to‑1
    tl.atomic_max(bitmap_ptr + orderkey, tl.full((1,), 1, tl.int32))
