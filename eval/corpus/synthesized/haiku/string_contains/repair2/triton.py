import triton
import triton.language as tl
import torch


@triton.jit
def kernel(offsets, chars, pattern, match, n, k, BLOCK_SIZE: tl.constexpr):
    """
    Check if pattern is contained in each string.
    offsets: int32 array of n+1 offsets
    chars: uint8 array of total characters
    pattern: uint8 array of pattern of length k
    match: int8 output array of n results
    """
    idx = tl.program_id(0) * BLOCK_SIZE

    for i in range(BLOCK_SIZE):
        pos = idx + i
        if pos < n:
            # Load string offsets with bounds checking
            start = tl.load(offsets + pos)
            # Ensure pos + 1 < n + 1 for loading offsets[pos + 1]
            is_in_bounds = pos + 1 < n + 1
            end = tl.where(is_in_bounds, tl.load(offsets + pos + 1), start)
            
            str_len = end - start
            
            # Handle empty pattern
            if k == 0:
                tl.store(match + pos, 1)
                continue
            
            # If string is too short, no match
            if str_len < k:
                tl.store(match + pos, 0)
                continue
            
            # Search for pattern
            found = 0
            for scan_idx in range(k):  # Limit to pattern length iterations for search
                all_match = 1
                # Check pattern match at this position
                for j in range(k):
                    char_pos = start + scan_idx + j
                    # Explicit bounds check before accessing chars
                    if char_pos >= 0 and char_pos < 1000000 and j >= 0 and j < k:
                        char_val = tl.load(chars + char_pos)
                        pattern_val = tl.load(pattern + j)
                        if char_val != pattern_val:
                            all_match = 0
                            break
                    else:
                        all_match = 0
                        break
                
                if all_match:
                    found = 1
                    break
            
            tl.store(match + pos, found)


TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "pattern": "*u8",
    "match": "*i8",
    "n": "i64",
    "k": "i64"
}
CONSTANTS = {"BLOCK_SIZE": 256}
NUM_WARPS = 4


def run(offsets, chars, pattern, match, n, k):
    """
    offsets: torch.tensor(shape=(n+1,), dtype=torch.int32)
    chars: torch.tensor(shape=(offsets[n],), dtype=torch.uint8)
    pattern: torch.tensor(shape=(k,), dtype=torch.uint8)
    match: torch.tensor(shape=(n,), dtype=torch.int8)
    n: int
    k: int
    """
    grid = ((n + CONSTANTS["BLOCK_SIZE"] - 1) // CONSTANTS["BLOCK_SIZE"],)
    kernel[grid](offsets, chars, pattern, match, n, k, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
