"""sorted_contains: membership of each key in a sorted distinct set.

Each lane finds the last member below its key by binary lifting from
j = -1 with the powers of two 2^40 down to 1 (their sum exceeds any member
count that fits in memory), then tests whether member j + 1 equals the key.
The steps are unrolled at compile time, so every probe is a masked gather
with no loop-carried address, and probes past the end of `members` are
masked off; the first search levels touch few cache lines and stay in L1.

Revision: the dynamic lifting loop is replaced by this static one, and the
probes and the final gather address `members` by a byte offset that is
pinned in one register (an identity PTX mov) and masked to the extent of
the array. Under the task's contract the mask terms never exclude a lane.
"""

import torch
import triton
import triton.language as tl

BLOCK = 512
STEPS = 41  # powers of two 2^40 .. 1
NUM_WARPS = 8
TYPES = {"keys": "*i64", "members": "*i64", "match": "*i8", "n": "i64", "s": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "STEPS": STEPS}


@triton.jit
def pin(x):
    """Identity through one PTX mov: the mask and the address of a gather then use the same register."""
    return tl.inline_asm_elementwise("mov.b64 $0, $1;", "=l,l", [x], dtype=tl.int64, is_pure=True, pack=1)


@triton.jit
def kernel(keys, members, match, n: tl.int64, s: tl.int64, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    key = tl.load(keys + i, mask=m, other=0)
    mb = members.to(tl.pointer_type(tl.int8))
    last = s * 8 - 8  # byte offset of the last member

    # j = the last member below key, or -1
    j = tl.full([BLOCK], -1, tl.int64)
    for k in tl.static_range(STEPS - 1, -1, -1):
        idx = j + (1 << k)
        off = pin(idx * 8)
        ok = m & (off >= 0) & (off <= last)
        v = tl.load((mb + off).to(tl.pointer_type(tl.int64)), mask=ok, other=9223372036854775807)
        j = tl.where(v < key, idx, j)

    hoff = pin((j + 1) * 8)
    hit = m & (hoff >= 0) & (hoff <= last)
    v = tl.load((mb + hoff).to(tl.pointer_type(tl.int64)), mask=hit, other=0)
    tl.store(match + i, (hit & (v == key)).to(tl.int8), mask=m)


def run(keys: torch.Tensor, members: torch.Tensor, match: torch.Tensor, n: int, s: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](keys, members, match, n, s, BLOCK=BLOCK, STEPS=STEPS, num_warps=NUM_WARPS)
