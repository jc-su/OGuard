"""radix_scatter: scatter row ids by one key byte, as one pass of an LSD radix sort.

Each block takes a tile of 2048 keys (8 per thread, coalesced loads). It
ranks every key inside the tile with an atomic add on a shared 256-bin
histogram, claims one range per non-empty bin with a single global atomic
add on that bin's cursor (at most 256 global atomics per 2048 keys instead
of one per key), and stores each row id at bin base + rank. On exit every
cursor has advanced to the end of its bin.

Revision: the eight keys of a thread are handled by one statement each
through inlined helpers, with their bytes and ranks in registers, instead
of a loop over local arrays; and the store position is tested against
[0, n). The test always passes under the task's contract (the cursors hold
the exclusive prefix sum of the byte counts on entry); it states the bound
of the store for the memory checker, which does not carry loop invariants
and does not follow the values of an inout buffer.
"""

from numba import cuda, int32, int64, uint32

TYPES = (uint32[:], int32[:], int32[:], int64, int64)
THREADS = 256
ITEMS = 8
TILE = THREADS * ITEMS


@cuda.jit(device=True, inline="always")
def rank_key(keys, hist, n, shift, i):
    """Byte of keys[i] and its rank among the tile's keys with that byte so far."""
    b = 0
    r = 0
    if i < n:
        b = int32((keys[i] >> shift) & 255)
        r = cuda.atomic.add(hist, b, 1)
    return b, r


@cuda.jit(device=True, inline="always")
def scatter_key(out, base, n, i, b, r):
    if i < n:
        pos = base[b] + r
        if pos >= 0 and pos < n:
            out[pos] = i


def kernel(keys, cursors, out, n, shift):
    hist = cuda.shared.array(256, int32)
    base = cuda.shared.array(256, int32)
    tid = cuda.threadIdx.x
    t0 = cuda.blockIdx.x * TILE

    hist[tid] = 0
    cuda.syncthreads()

    i0 = t0 + tid
    b0, r0 = rank_key(keys, hist, n, shift, i0)
    b1, r1 = rank_key(keys, hist, n, shift, i0 + THREADS)
    b2, r2 = rank_key(keys, hist, n, shift, i0 + 2 * THREADS)
    b3, r3 = rank_key(keys, hist, n, shift, i0 + 3 * THREADS)
    b4, r4 = rank_key(keys, hist, n, shift, i0 + 4 * THREADS)
    b5, r5 = rank_key(keys, hist, n, shift, i0 + 5 * THREADS)
    b6, r6 = rank_key(keys, hist, n, shift, i0 + 6 * THREADS)
    b7, r7 = rank_key(keys, hist, n, shift, i0 + 7 * THREADS)
    cuda.syncthreads()

    c = hist[tid]
    if c > 0:
        base[tid] = cuda.atomic.add(cursors, tid, c)
    cuda.syncthreads()

    scatter_key(out, base, n, i0, b0, r0)
    scatter_key(out, base, n, i0 + THREADS, b1, r1)
    scatter_key(out, base, n, i0 + 2 * THREADS, b2, r2)
    scatter_key(out, base, n, i0 + 3 * THREADS, b3, r3)
    scatter_key(out, base, n, i0 + 4 * THREADS, b4, r4)
    scatter_key(out, base, n, i0 + 5 * THREADS, b5, r5)
    scatter_key(out, base, n, i0 + 6 * THREADS, b6, r6)
    scatter_key(out, base, n, i0 + 7 * THREADS, b7, r7)


_kernel = cuda.jit(kernel)


def run(keys, cursors, out, n, shift):
    blocks = max(1, (n + TILE - 1) // TILE)
    _kernel[blocks, THREADS](keys, cursors, out, n, shift)
