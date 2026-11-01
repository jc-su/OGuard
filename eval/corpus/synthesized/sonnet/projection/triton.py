"""Triton kernel for projection.

out[i] = price[i] * (1 - discount[i]) for i in [0, n).
"""

import torch
import triton
import triton.language as tl

TYPES = {
    "price": "*fp64",
    "discount": "*fp64",
    "out": "*fp64",
    "n": "i64",
}

CONSTANTS = {"BLOCK": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(price, discount, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(price + offs, mask=mask)
    d = tl.load(discount + offs, mask=mask)
    tl.store(out + offs, p * (1.0 - d), mask=mask)


def run(price, discount, out, n):
    """Launch kernel on PyTorch CUDA tensors."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](price, discount, out, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
