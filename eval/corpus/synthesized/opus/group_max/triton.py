"""Per-group maximum of an int64 column with an int64 atomic max per row."""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(group, value, best, n, capacity, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    k = tl.load(group + i, mask=m, other=0)
    v = tl.load(value + i, mask=m)
    tl.atomic_max(best + k, v, mask=m, sem="relaxed")


TYPES = {"group": "*i32", "value": "*i64", "best": "*i64", "n": "i64", "capacity": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(
    group: torch.Tensor, value: torch.Tensor, best: torch.Tensor, n: int, capacity: int
) -> None:
    grid = ((n + CONSTANTS["BLOCK"] - 1) // CONSTANTS["BLOCK"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](group, value, best, n, capacity, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
