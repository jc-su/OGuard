"""hash_build: insert (key, row id) into an open-addressing table with linear probing.

Home slot = mix64(key) & (capacity - 1), where mix64 is the MurmurHash3 64-bit
finalizer, so nearby keys spread over the table. A thread reads a slot first
and issues the atomicCAS only on an empty slot; when the CAS succeeds the thread
owns the slot and writes the row id. Duplicate keys, if any, get separate
slots, which is the multimap semantics a join build needs.

table_keys must be filled with -1 on entry.
"""

from numba import cuda, int32, int64, uint64

THREADS = 256
MAX_BLOCKS = 2048


@cuda.jit(device=True)
def mix64(key):
    h = uint64(key)
    h ^= h >> uint64(33)
    h *= uint64(0xFF51AFD7ED558CCD)
    h ^= h >> uint64(33)
    h *= uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> uint64(33)
    return h


def kernel(keys, table_keys, table_rows, capacity, n):
    mask = capacity - 1
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        key = keys[i]
        slot = int64(mix64(key) & uint64(mask))
        while True:
            if table_keys[slot] == -1:
                if cuda.atomic.cas(table_keys, slot, -1, key) == -1:
                    table_rows[slot] = i
                    break
            slot = (slot + 1) & mask
        i += stride


TYPES = (int64[:], int64[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(keys, table_keys, table_rows, capacity, n):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](keys, table_keys, table_rows, capacity, n)
