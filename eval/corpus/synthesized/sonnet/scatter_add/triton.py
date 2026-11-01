import torch
import triton
import triton.language as tl


@triton.jit
def kernel(idx, values, out, m, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < m
    target = tl.load(idx + offs, mask=inb)
    v = tl.load(values + offs, mask=inb)
    tl.atomic_add(out + target, v, mask=inb, sem="relaxed")


TYPES = {"idx": "*i32", "values": "*fp64", "out": "*fp64", "m": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(idx: torch.Tensor, values: torch.Tensor, out: torch.Tensor, m: int, n: int) -> None:
    if m == 0:
        return
    grid = (triton.cdiv(m, CONSTANTS["BLOCK"]),)
    kernel[grid](idx, values, out, m, n, **CONSTANTS, num_warps=NUM_WARPS)
