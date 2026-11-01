"""Triton kernel for segmented_sum.

sums[j] = sum(values[offsets[j]:offsets[j+1]]) for j in [0, g). One
program is assigned per segment. Each program walks its segment in
chunks of BLOCK values, reducing each chunk with tl.sum, so segments
of very different sizes are handled without one thread per value.

Repair 1 (kept): start = tl.load(offsets + pid), end = tl.load(offsets
+ pid + 1) and the final tl.store(sums + pid, total) are masked on
`pid < g`, so a grid larger than g (which the checker must allow,
since it proves safety for any grid) leaves those programs reading
`other=0` and skipping their store.

Repair 2: the checker still refused a load into values -- inside the
per-segment walk, `offs = pos + arange(0, BLOCK)` is masked with
`m = offs < end`, but `end` is itself a value loaded from offsets, not
a value the checker can bound the way it bounds `pid < g`. So two
more things changed:

  * `pid` is clamped into [0, g - 1] (g floored at 1) before it is
    used to address offsets or sums at all, so `offsets + pid` and
    `offsets + pid + 1` are unconditionally inside offsets (an array
    of g + 1 elements), and `sums + pid` is unconditionally inside
    sums, independent of the `pid < g` mask;
  * `values` has no length argument of its own -- its length is
    offsets[g] by the task's contract -- so that bound is loaded
    directly (`offsets + g`, always a valid index into offsets' g + 1
    elements, since g < g + 1 needs no masking) and used to clamp
    every `offs` into [0, values_len - 1] (floored at 1) right before
    it forms the values address, on top of the existing
    `m = offs < end` mask that still decides which lanes contribute
    real data versus 0.0.
"""

import triton
import triton.language as tl

TYPES = {
    "offsets": "*i32",
    "values": "*fp64",
    "sums": "*fp64",
    "g": "i64",
}

CONSTANTS = {"BLOCK": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(offsets, values, sums, g, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    pid_mask = pid < g

    # g_pos/g_last give pid a clamp bound that is well-formed even for
    # g == 0; safe_pid + 1 then stays <= g_pos <= g + 1's last index.
    g_pos = tl.maximum(g, 1)
    g_last = g_pos - 1
    safe_pid = tl.minimum(tl.maximum(pid, 0), g_last)

    start = tl.load(offsets + safe_pid, mask=pid_mask, other=0).to(tl.int64)
    end = tl.load(offsets + safe_pid + 1, mask=pid_mask, other=0).to(tl.int64)

    # values' length is offsets[g] by the task's contract; offsets has
    # g + 1 elements, so index g is always valid without masking.
    values_len = tl.load(offsets + g).to(tl.int64)
    values_len_pos = tl.maximum(values_len, 1)
    values_len_last = values_len_pos - 1

    total = tl.zeros((), dtype=tl.float64)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK).to(tl.int64)
        m = offs < end
        safe_offs = tl.minimum(tl.maximum(offs, 0), values_len_last)
        v = tl.load(values + safe_offs, mask=m, other=0.0)
        total += tl.sum(v)
        pos += BLOCK

    tl.store(sums + safe_pid, total, mask=pid_mask)


def run(offsets, values, sums, g):
    """Launch kernel on PyTorch CUDA tensors."""
    if g == 0:
        return
    grid = (g,)
    kernel[grid](offsets, values, sums, g, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
