"""Triton kernel for scatter_by_offsets.

For each i with flags[i] != 0, writes out[offsets[i]] = i. offsets is
the exclusive prefix sum of flags, so distinct selected rows land at
distinct, already-known positions and no atomics are needed.

Repair 1 (kept): offs is built in 64-bit arithmetic throughout (pid *
BLOCK + arange, all as int64), so pid * BLOCK cannot overflow and
wrap into a value that still passes `offs < n` while addressing
outside flags/offsets, for any grid.

Repair 2: the checker still refused the masked store to
out[pos] -- `pos` is a value read from the offsets array, and while
the task's contract ties it to out's length (offsets is the exclusive
prefix sum of flags, so pos < total whenever flags[i] != 0), that is
a fact about the *relationship* between two buffers' contents, not
something derivable from pos's own range the way `offs < n` is. The
checker cannot see that relationship, so from its point of view pos
is an arbitrary int32 loaded from memory. `pos` is now clamped with
tl.maximum/tl.minimum into [0, total - 1] (total floored at 1, so the
clamp is well-formed even when total == 0) right before it forms the
store address. `write_mask` also now checks `pos` against
`[0, total)` explicitly, in addition to `flags[i] != 0`, so a
contract-violating pos can no longer select a stale/garbage address
in the block on the strength of `flags[i] != 0` alone; correctness
for contract-following inputs is unchanged, since such a pos is
always already inside [0, total).
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

    # n_pos/n_last give offs a clamp bound that is well-formed even for
    # n == 0.
    n_pos = tl.maximum(n, 1)
    n_last = n_pos - 1
    safe_offs = tl.minimum(tl.maximum(offs, 0), n_last)

    f = tl.load(flags + safe_offs, mask=mask, other=0)
    pos = tl.load(offsets + safe_offs, mask=mask, other=0).to(tl.int64)

    # total_pos/total_last give pos a clamp bound that is well-formed
    # even for total == 0 (out is then empty and write_mask below is
    # always false, since pos can never be < 0).
    total_pos = tl.maximum(total, 1)
    total_last = total_pos - 1
    safe_pos = tl.minimum(tl.maximum(pos, 0), total_last)

    write_mask = mask & (f != 0) & (pos >= 0) & (pos < total)
    tl.store(out + safe_pos, offs.to(tl.int32), mask=write_mask)


def run(flags, offsets, out, n, total):
    """Launch kernel on PyTorch CUDA tensors."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](
        flags, offsets, out, n, total, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS
    )
