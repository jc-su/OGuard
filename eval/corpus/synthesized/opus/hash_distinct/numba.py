"""hash_distinct: insert the distinct keys into an open-addressing table with linear
probing from slot h(key) & (capacity - 1), h(x) = (x * 0x9E3779B97F4A7C15) >> 32 on
the unsigned 64-bit value; table_rows[s] = smallest row with the key of slot s.

One thread per row. A slot changes once, from -1 to a key, by compare-and-swap, so
a key is never stored twice and never moves. A plain read of a slot can only be
stale by showing -1; the CAS then returns the key that is there. Each thread lowers
table_rows[s] with an atomic min, skipped when the slot already holds a row that
is not larger (the stored row only decreases). The table holds at least twice as
many slots as keys, so probing ends.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32
GOLDEN = 0x9E3779B97F4A7C15

TYPES = (
    types.int64[:],  # keys
    types.int64[:],  # table_keys
    types.int32[:],  # table_rows
    types.int64,  # n
    types.int64,  # capacity
)


def kernel(keys, table_keys, table_rows, n, capacity):
    mask = types.uint64(capacity - 1)
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        key = keys[i]
        h = (types.uint64(key) * types.uint64(GOLDEN)) >> types.uint64(32)
        slot = types.int64(h & mask)
        while True:
            cur = table_keys[slot]
            if cur == -1:
                cur = cuda.atomic.cas(table_keys, slot, -1, key)
                if cur == -1:
                    cur = key  # inserted here
            if cur == key:
                if table_rows[slot] > i:
                    cuda.atomic.min(table_rows, slot, types.int32(i))
                break
            slot = (slot + 1) & (capacity - 1)
        i += stride


_kernel = cuda.jit(kernel)


def run(keys, table_keys, table_rows, n, capacity):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](keys, table_keys, table_rows, n, capacity)
