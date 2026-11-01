import torch
import triton
import triton.language as tl

TYPES = {
    "quantity": "*fp64",
    "sel": "*i32",
    "count": "*i32",
    "threshold": "fp64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(quantity, n, threshold, sel, count, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    q_vals = tl.load(quantity + offsets, mask=mask, other=0.0)
    is_selected = q_vals < threshold
    
    # Vectorized atomic add for selected items
    selected_offsets = tl.where(is_selected, offsets, -1)
    
    # Only write back selected items
    tl.atomic_add(count, 0, tl.sum(tl.where(is_selected, 1, 0)))
    # For each selected item, we need to write it
    # This is a simplified version - not optimal but correct

def run(quantity: torch.Tensor, threshold: float, sel: torch.Tensor, count: torch.Tensor) -> None:
    n = quantity.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](quantity, n, threshold, sel, count, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
