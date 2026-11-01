import torch
import triton
import triton.language as tl


@triton.jit
def kernel(group, value, best, n, capacity, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    k = tl.load(group + offs, mask=inb)
    v = tl.load(value + offs, mask=inb)
    tl.atomic_max(best + k, v, mask=inb, sem="relaxed")


TYPES = {
    "group": "*i32",
    "value": "*i64",
    "best": "*i64",
    "n": "i64",
    "capacity": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(
    group: torch.Tensor,
    value: torch.Tensor,
    best: torch.Tensor,
    n: int,
    capacity: int,
) -> None:
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](group, value, best, n, capacity, **CONSTANTS, num_warps=NUM_WARPS)
