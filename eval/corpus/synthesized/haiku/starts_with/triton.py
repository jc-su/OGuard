import torch
import triton
import triton.language as tl

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "prefix": "*u8",
    "match": "*i8",
    "k": "i64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(offsets, chars, prefix, k, n, match, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    
    start = tl.load(offsets + idx, mask=mask, other=0)
    end = tl.load(offsets + idx + 1, mask=mask, other=0)
    
    length = end - start
    is_match = length >= k
    
    # Check prefix match
    for j in range(k):
        if is_match:
            char_val = tl.load(chars + start + j, mask=mask, other=0)
            prefix_val = tl.load(prefix + j)
            is_match = is_match & (char_val == prefix_val)
    
    result = tl.where(is_match & mask, tl.cast(1, tl.int8), tl.cast(0, tl.int8))
    tl.store(match + idx, result, mask=mask)

def run(offsets: torch.Tensor, chars: torch.Tensor, prefix: torch.Tensor, match: torch.Tensor) -> None:
    n = offsets.numel() - 1
    k = prefix.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](offsets, chars, prefix, k, n, match, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
