import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, sorted, out, n, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    key = tl.load(keys + offs, mask=inb)
    # Branchless binary search: pos is the length of the prefix of `sorted`
    # known to be less than key; step halves from the largest power of two <= s.
    step = s * 0 + 1
    while step * 2 <= s:
        step = step * 2
    pos = tl.zeros([BLOCK], dtype=tl.int64)
    while step > 0:
        cand = pos + step
        ok = inb & (cand <= s)
        probe = tl.load(sorted + cand - 1, mask=ok)
        pos = tl.where(ok & (probe < key), cand, pos)
        step = step >> 1
    tl.store(out + offs, pos.to(tl.int32), mask=inb)


TYPES = {"keys": "*i64", "sorted": "*i64", "out": "*i32", "n": "i64", "s": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 4


def run(
    keys: torch.Tensor, sorted: torch.Tensor, out: torch.Tensor, n: int, s: int
) -> None:
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, sorted, out, n, s, **CONSTANTS, num_warps=NUM_WARPS)
