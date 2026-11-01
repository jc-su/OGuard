import torch
import triton
import triton.language as tl

TYPES = {
    "offsets": "*i32",
    "values": "*fp64",
    "sums": "*fp64",
    "g": "i64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(offsets, values, g, sums, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < g
    
    seg_start = tl.load(offsets + idx, mask=mask, other=0)
    seg_end = tl.load(offsets + idx + 1, mask=mask, other=0)
    
    # Sum values in this segment
    seg_sum = 0.0
    for i in range(seg_start, seg_end):
        val = tl.load(values + i, mask=mask, other=0.0)
        seg_sum += val
    
    tl.store(sums + idx, seg_sum, mask=mask)

def run(offsets: torch.Tensor, values: torch.Tensor, sums: torch.Tensor) -> None:
    g = offsets.numel() - 1
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((g + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](offsets, values, g, sums, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
