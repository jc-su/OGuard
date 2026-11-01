"""hash_probe: emit (build row, probe row) for every probe key found in the table.

Each thread probes one row at a time with linear probing from
mix64(key) & (capacity - 1) and stops at the first empty slot, so a build
side with duplicate keys emits every match. Output writes are
warp-aggregated: on a match, the thread reads the active mask, the lowest
active lane reserves the slots for all active lanes with one atomicAdd on
out_count, and every active lane writes at base + (number of active lanes
below it). Under independent thread scheduling the active mask can cover a
subset of the lanes that hold a match; each subset then reserves its own
slots, which is still correct.

The store sits inside the row loop, whose condition i < m bounds the stored
probe row id on every path to the store. The output position is formed from
base as an unsigned value and compared against out_cap before the store, so
every store stays inside out_build and out_probe (out_cap slots) whatever
out_count holds on entry. Pairs past out_cap are counted but not written, so
the caller can size the output from out_count and probe again.

out_count must be zero on entry.
"""

from numba import cuda, int32, int64, uint32, uint64

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


def kernel(
    probe_keys, table_keys, table_rows, capacity, out_build, out_probe, out_count, m, out_cap
):
    lanemask_lt = cuda.lanemask_lt()
    mask = capacity - 1
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    while i < m:
        key = probe_keys[i]
        slot = int64(mix64(key) & uint64(mask))
        while True:
            tk = table_keys[slot]
            if tk == -1:
                break
            if tk == key:
                build_row = table_rows[slot]
                active = cuda.activemask()
                rank = cuda.popc(active & lanemask_lt)
                base = int32(0)
                if rank == 0:
                    base = cuda.atomic.add(out_count, 0, cuda.popc(active))
                base = cuda.shfl_sync(active, base, cuda.ffs(active) - 1)
                pos = int64(uint32(base)) + rank
                if pos < out_cap:
                    out_build[pos] = build_row
                    out_probe[pos] = i
            slot = (slot + 1) & mask
        i += stride


TYPES = (int64[:], int64[:], int32[:], int64, int32[:], int32[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(probe_keys, table_keys, table_rows, capacity, out_build, out_probe, out_count, m, out_cap):
    blocks = max(1, min((m + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](
        probe_keys, table_keys, table_rows, capacity, out_build, out_probe, out_count, m, out_cap
    )
