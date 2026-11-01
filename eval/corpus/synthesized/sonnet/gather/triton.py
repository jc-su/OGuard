"""Triton kernel for gather.

out[i] = col[sel[i]] for i in [0, m). Every value in sel is a row id
of col, in [0, n).
"""

import triton
import triton.language as tl

TYPES = {
    "col": "*fp64",
    "sel": "*i32",
    "out": "*fp64",
    "m": "i64",
    "n": "i64",
}

CONSTANTS = {"BLOCK": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(col, sel, out, m, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < m
    idx = tl.load(sel + offs, mask=mask, other=0)
    val = tl.load(col + idx, mask=mask)
    tl.store(out + offs, val, mask=mask)


def run(col, sel, out, m, n):
    """Launch kernel on PyTorch CUDA tensors."""
    if m == 0:
        return
    grid = (triton.cdiv(m, CONSTANTS["BLOCK"]),)
    kernel[grid](col, sel, out, m, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
