"""Triton kernel for histogram.

Counts keys by the byte (key >> shift) & 255, one pass of a radix
sort. Each program bins its own block of keys with tl.histogram (a
block-local histogram, computed without any per-key atomic), then
flushes the 256 bin counts to the global counts with one atomic add
per bin per program, instead of one atomic add per key.

Precondition: the caller zeroes counts before each launch.

Repair 1 (kept): offs is built in 64-bit arithmetic throughout (pid *
BLOCK + arange, all as int64), so pid * BLOCK cannot overflow and
wrap into a value that still passes `offs < n` while addressing
outside keys, for any grid.

Repair 2: the checker still refused the masked load of keys[offs],
even though `mask = offs < n` is exactly the bound needed. The
checker apparently does not correlate a predicate register with the
value used to build the address in the same instruction (it is not
enough that the two are related; the address value itself must carry
a directly checkable bound). So offs is now also clamped with
tl.maximum/tl.minimum into [0, n - 1] (n floored at 1, so the clamp
is well-formed even when n == 0) right before it forms the load
address, independent of the mask. The mask keeps gating which lanes
get real data versus `other=0`, exactly as before, so which keys are
counted is unchanged; only the address fed to the load instruction is
now unconditionally bounded. `counts + bin_offs` is unaffected (and
also given the same defensive clamp): bin_offs is `tl.arange(0, 256)`,
which already matches counts' fixed length exactly.
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

    # n_pos/n_last give offs a clamp bound that is well-formed even for
    # n == 0 (in which case the clamped address is never actually read
    # from, since mask is all-false); the mask, not the clamp, still
    # decides correctness.
    n_pos = tl.maximum(n, 1)
    n_last = n_pos - 1
    safe_offs = tl.minimum(tl.maximum(offs, 0), n_last)

    k = tl.load(keys + safe_offs, mask=mask, other=0)
    b = ((k >> shift) & 255).to(tl.int32)

    hist = tl.histogram(b, NUM_BINS, mask=mask)
    bin_offs = tl.arange(0, NUM_BINS)
    safe_bin_offs = tl.minimum(tl.maximum(bin_offs, 0), NUM_BINS - 1)
    tl.atomic_add(counts + safe_bin_offs, hist)


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
