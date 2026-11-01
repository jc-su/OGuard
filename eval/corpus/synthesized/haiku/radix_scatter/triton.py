import triton
import triton.language as tl
import torch


@triton.jit
def kernel(keys, cursors, out, n, shift, BLOCK_SIZE: tl.constexpr):
    """
    Scatter keys by byte for radix sort.
    keys: uint32 array of n keys
    cursors: int32 array of 256 cursors (exclusive prefix sum of byte counts)
    out: int32 output array of n row ids
    n: number of keys
    shift: shift amount (0 to 24, in multiples of 8)
    """
    idx = tl.program_id(0) * BLOCK_SIZE

    for i in range(BLOCK_SIZE):
        pos = idx + i
        if pos < n:
            # Get the key
            key = tl.load(keys + pos)

            # Extract the byte at the given shift
            byte_val = (key >> shift) & 255

            # Atomically increment the cursor for this byte
            cursor_idx = byte_val
            position = tl.atomic_add(cursors + cursor_idx, 1)

            # Store the row id at the position
            tl.store(out + position, pos, eviction_policy='evict_last')


TYPES = {
    "keys": "*u32",
    "cursors": "*i32",
    "out": "*i32",
    "n": "i64",
    "shift": "i64"
}
CONSTANTS = {"BLOCK_SIZE": 256}
NUM_WARPS = 4


def run(keys, cursors, out, n, shift):
    """
    keys: torch.tensor(shape=(n,), dtype=torch.uint32)
    cursors: torch.tensor(shape=(256,), dtype=torch.int32)
    out: torch.tensor(shape=(n,), dtype=torch.int32)
    n: int (number of keys)
    shift: int (shift amount, 0 to 24)
    """
    grid = ((n + CONSTANTS["BLOCK_SIZE"] - 1) // CONSTANTS["BLOCK_SIZE"],)
    kernel[grid](keys, cursors, out, n, shift, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
