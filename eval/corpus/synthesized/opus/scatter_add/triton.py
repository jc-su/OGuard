"""Add values[i] to out[idx[i]] with a float64 atomic add per row."""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(idx, values, out, m, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = i < m
    row = tl.load(idx + i, mask=live, other=0)
    v = tl.load(values + i, mask=live)
    tl.atomic_add(out + row, v, mask=live, sem="relaxed")


TYPES = {"idx": "*i32", "values": "*fp64", "out": "*fp64", "m": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(idx: torch.Tensor, values: torch.Tensor, out: torch.Tensor, m: int, n: int) -> None:
    grid = ((m + CONSTANTS["BLOCK"] - 1) // CONSTANTS["BLOCK"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](idx, values, out, m, n, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
