import triton
import triton.language as tl
import torch


@triton.jit
def kernel(offsets, chars, pattern, match, n, k, BLOCK_SIZE: tl.constexpr):
    """
    Find pattern in strings.
    offsets: int32 array of n+1 offset values
    chars: uint8 array of characters
    pattern: uint8 array of k pattern bytes
    match: int8 output array of n values
    n: number of strings
    k: pattern length
    """
    idx = tl.program_id(0) * BLOCK_SIZE

    for i in range(BLOCK_SIZE):
        string_id = idx + i
        if string_id < n:
            # Get start and end offsets for this string
            start_offset = tl.load(offsets + string_id)
            end_offset = tl.load(offsets + string_id + 1)
            string_len = end_offset - start_offset

            # Handle empty pattern (occurs in every string)
            if k == 0:
                tl.store(match + string_id, 1, eviction_policy='evict_last')
            else:
                # Search for pattern in the string
                found = 0
                # Try each starting position in the string
                for start_pos in range(string_len):
                    if start_pos + k <= string_len:
                        # Check if pattern matches at this position
                        match_found = 1
                        for p_idx in range(k):
                            char_in_string = tl.load(chars + start_offset + start_pos + p_idx)
                            pattern_char = tl.load(pattern + p_idx)
                            if char_in_string != pattern_char:
                                match_found = 0
                        if match_found:
                            found = 1

                tl.store(match + string_id, found, eviction_policy='evict_last')


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
    chars: torch.tensor(shape=(total_chars,), dtype=torch.uint8)
    pattern: torch.tensor(shape=(k,), dtype=torch.uint8)
    match: torch.tensor(shape=(n,), dtype=torch.int8)
    n: int (number of strings)
    k: int (pattern length)
    """
    grid = ((n + CONSTANTS["BLOCK_SIZE"] - 1) // CONSTANTS["BLOCK_SIZE"],)
    kernel[grid](offsets, chars, pattern, match, n, k, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
