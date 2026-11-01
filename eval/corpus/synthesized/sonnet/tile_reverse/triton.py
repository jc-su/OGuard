import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, out, n, TILE: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * TILE
    length = tl.minimum(n - base, TILE)
    lane = tl.arange(0, TILE)
    inb = lane < length
    v = tl.load(values + base + length - 1 - lane, mask=inb)
    tl.store(out + base + lane, v, mask=inb)


TYPES = {"values": "*i32", "out": "*i32", "n": "i64"}
CONSTANTS = {"TILE": 1024}
NUM_WARPS = 4


def run(values: torch.Tensor, out: torch.Tensor, n: int) -> None:
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["TILE"]),)
    kernel[grid](values, out, n, **CONSTANTS, num_warps=NUM_WARPS)
