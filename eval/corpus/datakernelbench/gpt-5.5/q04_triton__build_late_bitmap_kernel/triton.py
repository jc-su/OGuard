import triton
import triton.language as tl

TYPES = {
    "l_orderkey_ptr": "*i64",
    "l_commitdate_days_ptr": "*i32",
    "l_receiptdate_days_ptr": "*i32",
    "bitmap_ptr": "*u8",
    "n_elements": "i32",
    "bitmap_size": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_orderkey_ptr,
    l_commitdate_days_ptr,
    l_receiptdate_days_ptr,
    bitmap_ptr,
    n_elements,
    bitmap_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    orderkey = tl.load(l_orderkey_ptr + offs, mask=mask, other=-1)
    commitdate = tl.load(l_commitdate_days_ptr + offs, mask=mask, other=0)
    receiptdate = tl.load(l_receiptdate_days_ptr + offs, mask=mask, other=0)

    keep = mask & (commitdate < receiptdate) & (orderkey >= 0) & (orderkey < bitmap_size)
    one = tl.full((BLOCK_SIZE,), 1, tl.uint8)
    tl.store(bitmap_ptr + orderkey, one, mask=keep)
