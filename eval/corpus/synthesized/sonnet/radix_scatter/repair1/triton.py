import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(keys, cursors, out, n, shift, BLOCK_SIZE: tl.constexpr):
    # Compute the flat offset in 64-bit arithmetic: program_id and arange
    # are 32-bit by default, and for an adversarial (oversized) grid
    # `pid * BLOCK_SIZE` can overflow a 32-bit accumulator, defeating the
    # `offs < n` mask.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n

    key = tl.load(keys + offs, mask=mask, other=0)
    byte = ((key >> shift) & 255).to(tl.int32)

    # Per-lane scatter atomic: each lane claims its own slot in its byte's
    # bucket and advances that bucket's cursor in the same instruction.
    pos = tl.atomic_add(cursors + byte, 1, mask=mask).to(tl.int64)
    # `pos` is the pre-increment value of a bucket cursor, not a direct
    # function of the lane's offset; it is bounded by `n` only through the
    # global invariant that `cursors` starts as a correct histogram
    # prefix sum over every row (part of this task's contract), which is
    # not something a per-lane bound check can see. Guard the store
    # against `out`'s declared length directly so its address is provably
    # in range for any input.
    store_mask = mask & (pos >= 0) & (pos < n)
    tl.store(out + pos, offs.to(tl.int32), mask=store_mask)


TYPES = {
    "keys": "*u32",
    "cursors": "*i32",
    "out": "*i32",
    "n": "i64",
    "shift": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE}
NUM_WARPS = 8


def run(keys, cursors, out, n, shift):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](keys, cursors, out, n, shift, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
