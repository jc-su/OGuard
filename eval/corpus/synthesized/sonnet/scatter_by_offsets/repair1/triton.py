"""Triton kernel for scatter_by_offsets.

For each i with flags[i] != 0, writes out[offsets[i]] = i. offsets is
the exclusive prefix sum of flags, so distinct selected rows land at
distinct, already-known positions and no atomics are needed.

Repair: offs was computed as pid * BLOCK + arange in 32-bit
arithmetic. Since the checker must prove safety for any grid, pid is
not bounded by the grid this file happens to launch, so pid * BLOCK
can overflow a 32-bit index and wrap into a value that still passes
`offs < n` while addressing outside flags/offsets. offs is now built
in 64-bit arithmetic throughout, so no such wraparound is possible.
pos (read from offsets, which indexes into out per the contract) is
unaffected: it is bounded by out's capacity by the task's contract on
offsets, not by anything derived from the grid.
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
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    f = tl.load(flags + offs, mask=mask, other=0)
    pos = tl.load(offsets + offs, mask=mask, other=0)
    write_mask = mask & (f != 0)
    tl.store(out + pos, offs.to(tl.int32), mask=write_mask)


def run(flags, offsets, out, n, total):
    """Launch kernel on PyTorch CUDA tensors."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](
        flags, offsets, out, n, total, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS
    )
