import triton
import triton.language as tl
import torch


@triton.jit
def kernel(keys, members, match, n, s, BLOCK_SIZE: tl.constexpr):
    """
    Check membership in sorted array.
    keys: int64 array of n values
    members: int64 sorted array of s distinct values
    match: int8 output array of n values
    n: number of keys
    s: size of members
    """
    idx = tl.program_id(0) * BLOCK_SIZE

    for i in range(BLOCK_SIZE):
        pos = idx + i
        if pos < n:
            key = tl.load(keys + pos)

            # Binary search in members without using break
            left = 0
            right = s
            found = 0

            for _ in range(100):  # Maximum 100 iterations for binary search
                # Continue searching only if left < right
                if left < right:
                    mid = (left + right) // 2
                    mid_val = tl.load(members + mid)

                    if mid_val < key:
                        left = mid + 1
                    elif mid_val > key:
                        right = mid
                    else:
                        found = 1

            tl.store(match + pos, found, eviction_policy='evict_last')


TYPES = {
    "keys": "*i64",
    "members": "*i64",
    "match": "*i8",
    "n": "i64",
    "s": "i64"
}
CONSTANTS = {"BLOCK_SIZE": 256}
NUM_WARPS = 4


def run(keys, members, match, n, s):
    """
    keys: torch.tensor(shape=(n,), dtype=torch.int64)
    members: torch.tensor(shape=(s,), dtype=torch.int64), sorted and distinct
    match: torch.tensor(shape=(n,), dtype=torch.int8)
    n: int (number of keys)
    s: int (size of members)
    """
    grid = ((n + CONSTANTS["BLOCK_SIZE"] - 1) // CONSTANTS["BLOCK_SIZE"],)
    kernel[grid](keys, members, match, n, s, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
