import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(keys, cursors, out, n, shift, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    key = tl.load(keys + offs, mask=mask, other=0)
    byte = ((key >> shift) & 255).to(tl.int32)

    # Per-lane scatter atomic: each lane claims its own slot in its byte's
    # bucket and advances that bucket's cursor in the same instruction.
    pos = tl.atomic_add(cursors + byte, 1, mask=mask)
    tl.store(out + pos, offs.to(tl.int32), mask=mask)


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
