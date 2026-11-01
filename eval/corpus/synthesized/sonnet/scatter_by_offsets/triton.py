"""Triton kernel for scatter_by_offsets.

For each i with flags[i] != 0, writes out[offsets[i]] = i. offsets is
the exclusive prefix sum of flags, so distinct selected rows land at
distinct, already-known positions and no atomics are needed.
"""

import triton
import triton.language as tl

TYPES = {
    "flags": "*i8",
    "offsets": "*i32",
    "out": "*i32",
    "n": "i64",
    "total": "i64",
}

CONSTANTS = {"BLOCK": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(flags, offsets, out, n, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    f = tl.load(flags + offs, mask=mask, other=0)
    pos = tl.load(offsets + offs, mask=mask, other=0)
    write_mask = mask & (f != 0)
    tl.store(out + pos, offs, mask=write_mask)


def run(flags, offsets, out, n, total):
    """Launch kernel on PyTorch CUDA tensors."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](
        flags, offsets, out, n, total, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS
    )
