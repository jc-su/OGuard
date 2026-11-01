import triton
import triton.language as tl

TYPES = {
    "partkey_ptr": "*i64",
    "cost_ptr": "*fp64",
    "min_cost_ptr": "*fp64",
    "keep_mask_ptr": "*i32",
    "n": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 1024
}

NUM_WARPS = 4


@triton.jit
def kernel(
    partkey_ptr,
    cost_ptr,
    min_cost_ptr,
    keep_mask_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    pk = tl.load(partkey_ptr + offs, mask=mask, other=0)
    cost = tl.load(cost_ptr + offs, mask=mask, other=-1.0)
    min_c = tl.load(min_cost_ptr + pk)
    diff = tl.abs(cost - min_c)
    keep = mask & (diff < 1e-9)
    tl.store(keep_mask_ptr + offs, keep.to(tl.int32))
