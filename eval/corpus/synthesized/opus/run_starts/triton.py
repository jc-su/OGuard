"""Flag the first row of each run of equal int64 values."""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, flags, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(values + i, mask=m)
    prev = tl.load(values + i - 1, mask=m & (i > 0))
    f = (i == 0) | (v != prev)
    tl.store(flags + i, f.to(tl.int8), mask=m)


TYPES = {"values": "*i64", "flags": "*i8", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(values: torch.Tensor, flags: torch.Tensor, n: int) -> None:
    grid = ((n + CONSTANTS["BLOCK"] - 1) // CONSTANTS["BLOCK"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](values, flags, n, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
