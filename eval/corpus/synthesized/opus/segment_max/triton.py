"""Maximum of each segment of an int64 column given by int32 offsets.

One program reduces one segment, in chunks of BLOCK rows. An empty segment
gives the smallest int64 value.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(offsets, values, maxes, g, BLOCK: tl.constexpr):
    j = tl.program_id(0).to(tl.int64)
    lo = tl.load(offsets + j).to(tl.int64)
    hi = tl.load(offsets + j + 1).to(tl.int64)
    acc = tl.full((BLOCK,), -9223372036854775808, tl.int64)
    for start in range(lo, hi, BLOCK):
        k = start + tl.arange(0, BLOCK)
        x = tl.load(values + k, mask=k < hi, other=-9223372036854775808)
        acc = tl.maximum(acc, x)
    # ty rejects calls to Triton @jit functions by generic invariance; Triton accepts them.
    tl.store(maxes + j, tl.max(acc, axis=0))  # ty: ignore[invalid-argument-type]


TYPES = {"offsets": "*i32", "values": "*i64", "maxes": "*i64", "g": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 2


def run(offsets: torch.Tensor, values: torch.Tensor, maxes: torch.Tensor, g: int) -> None:
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[(g,)](offsets, values, maxes, g, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
