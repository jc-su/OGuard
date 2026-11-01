"""Triton kernel for histogram.

Counts keys by the byte (key >> shift) & 255, one pass of a radix
sort. Each program bins its own block of keys with tl.histogram (a
block-local histogram, computed without any per-key atomic), then
flushes the 256 bin counts to the global counts with one atomic add
per bin per program, instead of one atomic add per key.

Precondition: the caller zeroes counts before each launch.

Repair: offs was computed as pid * BLOCK + arange in 32-bit
arithmetic. Since the checker must prove safety for any grid, pid is
not bounded by the grid this file happens to launch, so pid * BLOCK
can overflow a 32-bit index and wrap into a value that still passes
`offs < n` while addressing outside keys. offs is now built in 64-bit
arithmetic throughout, so no such wraparound is possible. counts +
bin_offs is unaffected: bin_offs ranges over exactly [0, 256), which
matches counts' fixed length.
"""

import triton
import triton.language as tl

TYPES = {
    "keys": "*u32",
    "counts": "*i32",
    "shift": "i64",
    "n": "i64",
}

CONSTANTS = {"BLOCK": 4096, "NUM_BINS": 256}

NUM_WARPS = 4


@triton.jit
def kernel(keys, counts, shift, n, BLOCK: tl.constexpr, NUM_BINS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n

    k = tl.load(keys + offs, mask=mask, other=0)
    b = ((k >> shift) & 255).to(tl.int32)

    hist = tl.histogram(b, NUM_BINS, mask=mask)
    bin_offs = tl.arange(0, NUM_BINS)
    tl.atomic_add(counts + bin_offs, hist)


def run(keys, counts, shift, n):
    """Launch kernel on PyTorch CUDA tensors. counts must already be zeroed."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](
        keys,
        counts,
        shift,
        n,
        BLOCK=CONSTANTS["BLOCK"],
        NUM_BINS=CONSTANTS["NUM_BINS"],
        num_warps=NUM_WARPS,
    )
