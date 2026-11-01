"""groupby_sum: hash aggregation of value by group into an open-addressing table.

Home slot = mix64(group) & (capacity - 1) with linear probing. A row first
reads the slot: a matching key gets an atomicAdd on the sum; an empty slot is
claimed with atomicCAS (a lost race against the same key still lands on the
right slot); any other key moves to the next slot. Keys never change once set,
so the plain read before the CAS is safe.

When capacity fits in shared memory (capacity <= SMEM_CAP) each block first
aggregates into a block-private copy of the table, then flushes its non-empty
slots into the global table with the same insert routine. This turns n global
atomics into at most capacity per block. Larger tables use the global table
directly.

table_keys must be filled with -1 and table_sums with zero on entry.
"""

from numba import cuda, float64, int32, int64, uint64

THREADS = 256
MAX_BLOCKS = 2048
SMEM_CAP = 2048  # 2048 * (4 + 8) bytes = 24 KiB of shared memory per block


@cuda.jit(device=True)
def home_slot(group, mask):
    h = uint64(int64(group))
    h ^= h >> uint64(33)
    h *= uint64(0xFF51AFD7ED558CCD)
    h ^= h >> uint64(33)
    h *= uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> uint64(33)
    return int64(h & uint64(mask))


@cuda.jit(device=True)
def insert_add(keys, sums, group, v, mask):
    slot = home_slot(group, mask)
    while True:
        cur = keys[slot]
        if cur == group:
            cuda.atomic.add(sums, slot, v)
            return
        if cur == -1:
            old = cuda.atomic.cas(keys, slot, -1, group)
            if old == -1 or old == group:
                cuda.atomic.add(sums, slot, v)
                return
        slot = (slot + 1) & mask


def kernel(group, value, table_keys, table_sums, capacity, n):
    mask = capacity - 1
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    if capacity <= SMEM_CAP:
        local_keys = cuda.shared.array(SMEM_CAP, int32)
        local_sums = cuda.shared.array(SMEM_CAP, float64)
        j = cuda.threadIdx.x
        while j < capacity:
            local_keys[j] = -1
            local_sums[j] = 0.0
            j += cuda.blockDim.x
        cuda.syncthreads()
        while i < n:
            insert_add(local_keys, local_sums, group[i], value[i], mask)
            i += stride
        cuda.syncthreads()
        j = cuda.threadIdx.x
        while j < capacity:
            g = local_keys[j]
            if g != -1:
                insert_add(table_keys, table_sums, g, local_sums[j], mask)
            j += cuda.blockDim.x
    else:
        while i < n:
            insert_add(table_keys, table_sums, group[i], value[i], mask)
            i += stride


TYPES = (int32[:], float64[:], int32[:], float64[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(group, value, table_keys, table_sums, capacity, n):
    # With block-private tables, a small grid gives each block many rows per
    # flushed slot; 256 blocks still fill every SM.
    max_blocks = 256 if capacity <= SMEM_CAP else MAX_BLOCKS
    blocks = max(1, min((n + THREADS - 1) // THREADS, max_blocks))
    _kernel[blocks, THREADS](group, value, table_keys, table_sums, capacity, n)
