import torch
import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, out, n, k, BLOCK: tl.constexpr, KMAX: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < n
    start = tl.load(offsets + rows, mask=rmask, other=0).to(tl.int64)
    end = tl.load(offsets + rows + 1, mask=rmask, other=0).to(tl.int64)
    take = tl.minimum(end - start, k)
    col = tl.arange(0, KMAX)[None, :]
    data = tl.load(
        chars + start[:, None] + col,
        mask=rmask[:, None] & (col < take[:, None]),
        other=0,
    )
    tl.store(out + rows[:, None] * k + col, data, mask=rmask[:, None] & (col < k))


TYPES = {"offsets": "*i32", "chars": "*u8", "out": "*u8", "n": "i64", "k": "i64"}
CONSTANTS = {"BLOCK": 32, "KMAX": 64}
NUM_WARPS = 4


def run(offsets, chars, out, n, k):
    assert 1 <= k <= CONSTANTS["KMAX"]
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](offsets, chars, out, n, k, **CONSTANTS, num_warps=NUM_WARPS)
