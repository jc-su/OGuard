import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    cur = tl.load(values + offs, mask=inb)
    prev = tl.load(values + offs - 1, mask=inb & (offs > 0), other=0)
    tl.store(out + offs, cur - prev, mask=inb)


TYPES = {"values": "*i64", "out": "*i64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(values: torch.Tensor, out: torch.Tensor, n: int) -> None:
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, out, n, **CONSTANTS, num_warps=NUM_WARPS)
