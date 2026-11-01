import torch
import triton
import triton.language as tl

TYPES = {
    "col": "*fp64",
    "sel": "*i32",
    "out": "*fp64",
    "m": "i64",
    "n": "i64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(col, sel, m, n, out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m
    
    sel_vals = tl.load(sel + offsets, mask=mask, other=0)
    col_vals = tl.load(col + sel_vals, mask=mask, other=0.0)
    
    tl.store(out + offsets, col_vals, mask=mask)

def run(col: torch.Tensor, sel: torch.Tensor, out: torch.Tensor) -> None:
    m = sel.numel()
    n = col.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((m + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](col, sel, m, n, out, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
