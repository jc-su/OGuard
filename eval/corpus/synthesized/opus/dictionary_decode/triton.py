"""Dictionary decode: out[i] = dictionary[codes[i]].

A streaming gather. The codes and the output are read and written once, so they
are marked evict-first, which leaves the L2 cache to the dictionary that every
program reads. Every code is in [0, d) by contract.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(codes, dictionary, out, n, d, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    code = tl.load(codes + offs, mask=in_range, other=0, eviction_policy="evict_first")
    value = tl.load(dictionary + code, mask=in_range)
    tl.store(out + offs, value, mask=in_range, eviction_policy="evict_first")


TYPES = {"codes": "*i32", "dictionary": "*fp64", "out": "*fp64", "n": "i64", "d": "i64"}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 8


def run(codes: torch.Tensor, dictionary: torch.Tensor, out: torch.Tensor, n: int, d: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](codes, dictionary, out, n, d, num_warps=NUM_WARPS, **CONSTANTS)
