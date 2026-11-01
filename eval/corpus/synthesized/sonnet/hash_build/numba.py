"""Numba CUDA kernel for hash_build.

Inserts each row of keys into an open-addressing hash table with
linear probing. capacity is a power of two, at least 2 * n, so the
table never fills and every probe sequence is guaranteed to reach an
empty slot.

Precondition: the caller fills table_keys with -1 before each launch.
"""

from numba import cuda, int32, int64, uint64

TYPES = (int64[:], int64[:], int32[:], int64, int64)

BLOCK = 256

EMPTY = -1


@cuda.jit(device=True, inline=True)
def _hash64(key):
    """splitmix64 finalizer: mixes a 64-bit key for table placement."""
    h = uint64(key)
    h = (h ^ (h >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> uint64(27))) * uint64(0x94D049BB133111EB)
    h = h ^ (h >> uint64(31))
    return h


def kernel(keys, table_keys, table_rows, capacity, n):
    i = cuda.grid(1)
    if i < n:
        key = keys[i]
        mask = capacity - 1
        slot = int64(_hash64(key) & uint64(mask))
        while True:
            old = cuda.atomic.cas(table_keys, slot, EMPTY, key)
            if old == EMPTY:
                table_rows[slot] = i
                break
            slot = (slot + 1) & mask


def run(keys, table_keys, table_rows, capacity, n):
    """Launch kernel on device arrays. table_keys must already be -1."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](keys, table_keys, table_rows, capacity, n)
