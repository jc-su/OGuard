import triton
import triton.language as tl

TYPES = {
    "o_orderkey_ptr": "*i64",
    "o_orderdate_ptr": "*i32",
    "o_priority_ptr": "*i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "bitmap_ptr": "*i32",
    "out_counts_ptr": "*i64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    o_orderkey_ptr,          # int64 *
    o_orderdate_ptr,         # int32 *
    o_priority_ptr,          # int32 *
    date_lo,                 # int32 scalar
    date_hi,                 # int32 scalar
    bitmap_ptr,              # int32 *
    out_counts_ptr,          # int64 * (size = num_priorities)
    n_elements,              # total rows in orders
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    orderdate = tl.load(o_orderdate_ptr + offs, mask=mask, other=0)
    keep_date = mask & (orderdate >= date_lo) & (orderdate < date_hi)

    orderkey = tl.load(o_orderkey_ptr + offs, mask=keep_date, other=0)

    bitmap_val = tl.load(bitmap_ptr + orderkey, mask=keep_date, other=0)
    keep = keep_date & (bitmap_val > 0)

    priority = tl.load(o_priority_ptr + offs, mask=keep, other=0)

    inc = keep.to(tl.int64)   # 1 where matched, 0 otherwise
    tl.atomic_add(out_counts_ptr + priority, inc)
