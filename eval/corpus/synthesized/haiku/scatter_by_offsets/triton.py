import torch
import triton
import triton.language as tl

TYPES = {
    "flags": "*i8",
    "offsets": "*i32",
    "out": "*i32",
    "n": "i64",
    "total": "i64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(flags, offsets, n, total, out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    
    flag_vals = tl.load(flags + idx, mask=mask, other=0)
    offset_vals = tl.load(offsets + idx, mask=mask, other=-1)
    
    is_set = flag_vals != 0
    if is_set:
        tl.store(out + offset_vals, tl.cast(idx, tl.int32), mask=mask)

def run(flags: torch.Tensor, offsets: torch.Tensor, total: int, out: torch.Tensor) -> None:
    n = flags.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](flags, offsets, n, total, out, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
