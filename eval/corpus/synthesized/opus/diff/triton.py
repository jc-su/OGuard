"""First difference of an int64 column: out[0] = values[0], out[i] = values[i] - values[i - 1]."""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(values + i, mask=m)
    prev = tl.load(values + i - 1, mask=m & (i > 0), other=0)
    tl.store(out + i, v - prev, mask=m)


TYPES = {"values": "*i64", "out": "*i64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(values: torch.Tensor, out: torch.Tensor, n: int) -> None:
    grid = ((n + CONSTANTS["BLOCK"] - 1) // CONSTANTS["BLOCK"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](values, out, n, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
