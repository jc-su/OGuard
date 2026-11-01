"""Sum of the float64 rows whose int8 mask is nonzero, added to result[0].

Each program reduces BLOCK rows and adds its partial sum with one atomic add.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, mask, result, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(values + i, mask=m, other=0.0)
    keep = tl.load(mask + i, mask=m, other=0)
    # ty rejects calls to Triton @jit functions by generic invariance; Triton accepts them.
    total = tl.sum(tl.where(keep != 0, x, 0.0), axis=0)  # ty: ignore[invalid-argument-type]
    tl.atomic_add(result, total, sem="relaxed")


TYPES = {"values": "*fp64", "mask": "*i8", "result": "*fp64", "n": "i64"}
CONSTANTS = {"BLOCK": 4096}
NUM_WARPS = 8


def run(values: torch.Tensor, mask: torch.Tensor, result: torch.Tensor, n: int) -> None:
    grid = ((n + CONSTANTS["BLOCK"] - 1) // CONSTANTS["BLOCK"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](values, mask, result, n, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
