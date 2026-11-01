import triton
import triton.language as tl

TYPES = {
    "table_ptr": "*i64",
    "counts_ptr": "*i64",
    "table_size": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    table_ptr,
    counts_ptr,
    table_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < table_size
    key = tl.load(table_ptr + offs, mask=mask, other=-1)
    keep = mask & (key != -1)

    gid = key >> 32
    one = tl.full((BLOCK_SIZE,), 1, tl.int64)
    tl.atomic_add(counts_ptr + gid, one, sem="relaxed", mask=keep)
