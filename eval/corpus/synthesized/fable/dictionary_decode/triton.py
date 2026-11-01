"""dictionary_decode: gather dictionary values by code, one row per lane."""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
TYPES = {"codes": "*i32", "dictionary": "*fp64", "out": "*fp64", "n": "i64", "d": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(codes, dictionary, out, n: tl.int64, d: tl.int64, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    c = tl.load(codes + i, mask=m, other=0)
    v = tl.load(dictionary + c, mask=m, other=0.0)
    tl.store(out + i, v, mask=m)


def run(codes: torch.Tensor, dictionary: torch.Tensor, out: torch.Tensor, n: int, d: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](codes, dictionary, out, n, d, BLOCK=BLOCK, num_warps=NUM_WARPS)
