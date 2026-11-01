"""run_end_decode: expand an Arrow run-end encoded column.

Each lane owns one output row i and finds the last run j whose end is at
most i by binary lifting over run_ends from j = -1 with the powers of two
2^31 down to 1 (run ends are int32, so r < 2^31); row i then belongs to run
j + 1. The steps are unrolled at compile time, so every probe is a masked
gather with no loop-carried address, and probes past the end of run_ends
are masked off. Rows of one program map to a contiguous range of runs, so
the probes hit few cache lines.

Revision: the dynamic lifting loop is replaced by this static one, and the
probes and the final gather address their arrays by a byte offset that is
pinned in one register (an identity PTX mov) and masked to the extent of
the array. Under the task's contract the mask terms never exclude a lane.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
STEPS = 32  # powers of two 2^31 .. 1: their sum exceeds any int32 run count
NUM_WARPS = 4
TYPES = {"run_values": "*fp64", "run_ends": "*i32", "out": "*fp64", "r": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "STEPS": STEPS}


@triton.jit
def pin(x):
    """Identity through one PTX mov: the mask and the address of a gather then use the same register."""
    return tl.inline_asm_elementwise("mov.b64 $0, $1;", "=l,l", [x], dtype=tl.int64, is_pure=True, pack=1)


@triton.jit
def kernel(run_values, run_ends, out, r: tl.int64, n: tl.int64, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    row = i.to(tl.int32)
    eb = run_ends.to(tl.pointer_type(tl.int8))
    vb = run_values.to(tl.pointer_type(tl.int8))
    last_end = r * 4 - 4  # byte offset of the last run end
    last_value = r * 8 - 8

    # j = the last run whose end is <= i, or -1
    j = tl.full([BLOCK], -1, tl.int64)
    for k in tl.static_range(STEPS - 1, -1, -1):
        idx = j + (1 << k)
        off = pin(idx * 4)
        ok = m & (off >= 0) & (off <= last_end)
        end = tl.load((eb + off).to(tl.pointer_type(tl.int32)), mask=ok, other=2147483647)
        j = tl.where(end <= row, idx, j)

    voff = pin((j + 1) * 8)
    okv = m & (voff >= 0) & (voff <= last_value)
    v = tl.load((vb + voff).to(tl.pointer_type(tl.float64)), mask=okv, other=0.0)
    tl.store(out + i, v, mask=m)


def run(run_values: torch.Tensor, run_ends: torch.Tensor, out: torch.Tensor, r: int, n: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](run_values, run_ends, out, r, n, BLOCK=BLOCK, STEPS=STEPS, num_warps=NUM_WARPS)
