# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(keys_ptr, part_ptr, counts_ptr, n, p):
    # Each program processes one key
    idx = tl.program_id(0)

    if idx < n:
        # Load the key
        key = tl.load(keys_ptr + idx)

        # Compute hash: h(x) = (x * 0x9E3779B97F4A7C15) >> 32
        # Need to be careful with unsigned arithmetic
        # Convert to unsigned
        ukey = tl.cast(key, tl.uint64)

        # Multiply by the magic constant
        magic = 0x9E3779B97F4A7C15
        hashed = ukey * magic

        # Right shift by 32 (keep upper 32 bits)
        hashed = hashed >> 32

        # Convert back to signed int64 for modulo operation
        hashed_i64 = tl.cast(hashed, tl.int64)

        # Compute partition index
        partition = hashed_i64 % p

        # Ensure partition is in [0, p)
        if partition < 0:
            partition = partition + p

        # Store partition index
        tl.store(part_ptr + idx, partition)

        # Atomically increment count for this partition
        tl.atomic_add(counts_ptr + partition, 1)


TYPES = {
    "keys_ptr": "*i64",
    "part_ptr": "*i32",
    "counts_ptr": "*i32",
    "n": "i64",
    "p": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(keys: torch.Tensor, part: torch.Tensor, counts: torch.Tensor, n: int, p: int) -> None:
    """
    Launch the hash_partition kernel.

    Args:
        keys: input keys (int64, n)
        part: output partition indices (int32, n)
        counts: output counts (int32, p), zero on entry
        n: number of keys
        p: number of partitions
    """
    if n == 0:
        return

    grid = (n,)
    kernel[grid](keys, part, counts, n, p, num_warps=NUM_WARPS)
