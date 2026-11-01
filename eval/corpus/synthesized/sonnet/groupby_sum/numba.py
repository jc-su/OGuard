"""Numba CUDA kernel for groupby_sum.

Hash aggregation: sums value per group into an open-addressing table
(table_keys, table_sums) with linear probing. capacity is a power of
two, at least 2 * (number of distinct groups), so the table never
fills.

Precondition: the caller fills table_keys with -1 and zeroes
table_sums before each launch.

Concurrent insertion is race free: two threads racing to claim the
same empty slot for the same group both see the winner's key (via
compare_and_swap or an equal already-installed key) and both add their
value; threads with different, colliding groups simply retry at the
next slot.
"""

from numba import cuda, float64, int32, int64, uint32, uint64

TYPES = (int32[:], float64[:], int32[:], float64[:], int64, int64)

BLOCK = 256

EMPTY = -1


@cuda.jit(device=True, inline=True)
def _hash32(key):
    """Murmur3 finalizer: mixes a 32-bit key for table placement."""
    h = uint32(key)
    h = h ^ (h >> uint32(16))
    h = h * uint32(0x85EBCA6B)
    h = h ^ (h >> uint32(13))
    h = h * uint32(0xC2B2AE35)
    h = h ^ (h >> uint32(16))
    return h


def kernel(group, value, table_keys, table_sums, capacity, n):
    i = cuda.grid(1)
    if i >= n:
        return

    g = group[i]
    v = value[i]
    mask = capacity - 1
    slot = int64(uint64(_hash32(g)) & uint64(mask))

    while True:
        old = cuda.atomic.cas(table_keys, slot, EMPTY, g)
        if old == EMPTY or old == g:
            cuda.atomic.add(table_sums, slot, v)
            break
        slot = (slot + 1) & mask


def run(group, value, table_keys, table_sums, capacity, n):
    """Launch kernel on device arrays.

    table_keys must already be -1 and table_sums must already be zero.
    """
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](group, value, table_keys, table_sums, capacity, n)
