import torch
import triton
import triton.language as tl

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "lengths": "*i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(offsets, chars, n, lengths, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    
    start = tl.load(offsets + idx, mask=mask, other=0)
    end = tl.load(offsets + idx + 1, mask=mask, other=0)
    
    # Count UTF-8 code points
    count = 0
    pos = start
    for _ in range(end - start):
        if pos >= end:
            break
        byte = tl.load(chars + pos, mask=mask, other=0)
        if byte < 128:
            count += 1
            pos += 1
        elif byte < 224:
            pos += 2
            count += 1
        elif byte < 240:
            pos += 3
            count += 1
        else:
            pos += 4
            count += 1
    
    tl.store(lengths + idx, count, mask=mask)

def run(offsets: torch.Tensor, chars: torch.Tensor, lengths: torch.Tensor) -> None:
    n = offsets.numel() - 1
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](offsets, chars, n, lengths, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
