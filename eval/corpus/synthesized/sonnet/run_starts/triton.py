import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, flags, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    cur = tl.load(values + offs, mask=inb)
    prev = tl.load(values + offs - 1, mask=inb & (offs > 0))
    start = (offs == 0) | (cur != prev)
    tl.store(flags + offs, start.to(tl.int8), mask=inb)


TYPES = {"values": "*i64", "flags": "*i8", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(values: torch.Tensor, flags: torch.Tensor, n: int) -> None:
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, flags, n, **CONSTANTS, num_warps=NUM_WARPS)
