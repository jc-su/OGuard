import torch
import triton
import triton.language as tl

TYPES = {
    "keys": "*u32",
    "counts": "*i32",
    "shift": "i64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(keys, n, shift, counts, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    
    key = tl.load(keys + idx, mask=mask, other=0)
    
    # Extract byte
    byte_val = (key >> shift) & 255
    
    tl.atomic_add(counts + byte_val, 1, mask=mask)

def run(keys: torch.Tensor, shift: int, counts: torch.Tensor) -> None:
    n = keys.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](keys, n, shift, counts, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
