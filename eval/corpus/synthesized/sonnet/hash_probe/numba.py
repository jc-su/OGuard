"""Numba CUDA kernel for hash_probe.

Probes the open-addressing hash table built by hash_build (table_keys,
table_rows, capacity) with probe_keys, and for each match emits the
pair (build row id, probe row id) into out_build / out_probe.

Precondition: the caller zeroes out_count before each launch.

Build keys are distinct, so at most one build row matches each probe
key: the probe stops at the first match. An empty slot (table_keys ==
-1) is reached within at most capacity steps, because capacity is at
least 2 * (number of build keys), so the table is never full.
"""

from numba import cuda, int32, int64, uint64

TYPES = (
    int64[:],
    int64[:],
    int32[:],
    int32[:],
    int32[:],
    int32[:],
    int64,
    int64,
    int64,
)

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


def kernel(
    probe_keys,
    table_keys,
    table_rows,
    out_build,
    out_probe,
    out_count,
    capacity,
    m,
    out_cap,
):
    j = cuda.grid(1)
    if j >= m:
        return

    key = probe_keys[j]
    mask = capacity - 1
    slot = int64(_hash64(key) & uint64(mask))

    while table_keys[slot] != EMPTY:
        if table_keys[slot] == key:
            pos = cuda.atomic.add(out_count, 0, 1)
            if pos < out_cap:
                out_build[pos] = table_rows[slot]
                out_probe[pos] = j
            break
        slot = (slot + 1) & mask


def run(
    probe_keys,
    table_keys,
    table_rows,
    out_build,
    out_probe,
    out_count,
    capacity,
    m,
    out_cap,
):
    """Launch kernel on device arrays. out_count must already be zeroed."""
    blocks = (m + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](
        probe_keys,
        table_keys,
        table_rows,
        out_build,
        out_probe,
        out_count,
        capacity,
        m,
        out_cap,
    )
