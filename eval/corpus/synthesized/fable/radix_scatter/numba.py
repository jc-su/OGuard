"""radix_scatter: scatter row ids by one key byte, as one pass of an LSD radix sort.

Each block takes a tile of 2048 keys (8 per thread, coalesced loads). It
ranks every key inside the tile with an atomic add on a shared 256-bin
histogram, claims one range per non-empty bin with a single global atomic
add on that bin's cursor (at most 256 global atomics per 2048 keys instead
of one per key), and stores each row id at bin base + rank. On exit every
cursor has advanced to the end of its bin.
"""

from numba import cuda, int32, int64, uint32

TYPES = (uint32[:], int32[:], int32[:], int64, int64)
THREADS = 256
ITEMS = 8
TILE = THREADS * ITEMS


def kernel(keys, cursors, out, n, shift):
    hist = cuda.shared.array(256, int32)
    base = cuda.shared.array(256, int32)
    ranks = cuda.local.array(ITEMS, int32)
    bins = cuda.local.array(ITEMS, int32)
    tid = cuda.threadIdx.x
    t0 = cuda.blockIdx.x * TILE

    hist[tid] = 0
    cuda.syncthreads()

    for k in range(ITEMS):
        i = t0 + k * THREADS + tid
        if i < n:
            b = (keys[i] >> shift) & 255
            bins[k] = b
            ranks[k] = cuda.atomic.add(hist, b, 1)
    cuda.syncthreads()

    c = hist[tid]
    if c > 0:
        base[tid] = cuda.atomic.add(cursors, tid, c)
    cuda.syncthreads()

    for k in range(ITEMS):
        i = t0 + k * THREADS + tid
        if i < n:
            out[base[bins[k]] + ranks[k]] = i


_kernel = cuda.jit(kernel)


def run(keys, cursors, out, n, shift):
    blocks = max(1, (n + TILE - 1) // TILE)
    _kernel[blocks, THREADS](keys, cursors, out, n, shift)
