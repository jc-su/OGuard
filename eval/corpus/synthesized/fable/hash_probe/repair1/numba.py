"""hash_probe: emit (build row, probe row) for every probe key found in the table.

Each lane probes one row with linear probing from mix64(key) & (capacity - 1)
and stops at the first empty slot, so a build side with duplicate keys emits
every match. Output writes are warp-aggregated: on every probe step the warp
ballots its matches, lane 0 reserves the slots with one atomicAdd on out_count,
and matching lanes write at base + rank. The probe loop runs in lockstep for
the whole warp (the row loop is rounded up to a multiple of the warp size) so
every ballot is a full-warp ballot.

The output position is formed from base as an unsigned value and compared
against out_cap before the store, so every store stays inside out_build and
out_probe (out_cap slots) whatever out_count holds on entry. Pairs past
out_cap are counted but not written, so the caller can size the output from
out_count and probe again.

out_count must be zero on entry.
"""

from numba import cuda, int32, int64, uint32, uint64

FULL_MASK = 0xFFFFFFFF
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
    lane = cuda.laneid
    mask = capacity - 1
    stride = cuda.gridsize(1)
    m_round = (m + 31) & ~31
    i = cuda.grid(1)
    while i < m_round:
        key = int64(0)
        slot = int64(0)
        done = True
        if i < m:
            key = probe_keys[i]
            slot = int64(mix64(key) & uint64(mask))
            done = False
        while True:
            found = False
            build_row = int32(0)
            if not done:
                tk = table_keys[slot]
                if tk == -1:
                    done = True
                elif tk == key:
                    found = True
                    build_row = table_rows[slot]
                slot = (slot + 1) & mask
            ballot = cuda.ballot_sync(FULL_MASK, found)
            if ballot != 0:
                base = int32(0)
                if lane == 0:
                    base = cuda.atomic.add(out_count, 0, cuda.popc(ballot))
                base = cuda.shfl_sync(FULL_MASK, base, 0)
                if found:
                    pos = int64(uint32(base)) + cuda.popc(ballot & cuda.lanemask_lt())
                    if pos < out_cap:
                        out_build[pos] = build_row
                        out_probe[pos] = i
            if cuda.ballot_sync(FULL_MASK, not done) == 0:
                break
        i += stride


TYPES = (int64[:], int64[:], int32[:], int64, int32[:], int32[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(probe_keys, table_keys, table_rows, capacity, out_build, out_probe, out_count, m, out_cap):
    blocks = max(1, min((m + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](
        probe_keys, table_keys, table_rows, capacity, out_build, out_probe, out_count, m, out_cap
    )
