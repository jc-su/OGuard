"""merge_sorted: merge two ascending int64 arrays, a before b on ties (Numba CUDA).

A merge-path merge. Each block produces the tile out[k0:k0 + tile_len] of at
most TILE outputs:

1. Split: find how many of the first k0 (and k1 = k0 + tile_len) outputs come
   from a, by a binary search along the cross diagonal (lane 0 of warp 0 for
   k0, lane 0 of warp 1 for k1, in parallel). With ties the split takes a
   first (a[i] <= b[k - 1 - i]). The probes of a binary search close in on
   the split, near data the block reads next; a 256-ary search with one probe
   per thread measured 3x slower, as its first rounds read hundreds of
   scattered sectors per block.
2. Load the tile's slices of a and b into shared memory with coalesced reads.
3. Each thread finds the split of its own diagonal (ITEMS outputs apart) in
   shared memory and merges ITEMS outputs sequentially into registers.
4. The merged tile goes back through shared memory so the stores coalesce.

ITEMS is odd so that the strided shared-memory accesses of step 4 have no
bank conflicts.

Bounds: every index is tested or clamped where it is used, against the length
of its own array, so no bound depends on a loop invariant or on the splits
that step 1 leaves in shared memory. A search probe into a or b is clamped to
[0, m - 1] or [0, p - 1] (the search runs only when m > 0 and p > 0, and it
finds nothing to search otherwise); a tile load tests its index against m or
p; a shared-memory index is clamped to [0, TILE - 1]; an output position is
compared with m + p. None of these tests changes a result for input that
meets the contract. The loops that index memory are while loops with the
bound in their condition.
"""

import numpy as np
from numba import cuda, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
ITEMS = 7
TILE = BLOCK * ITEMS
LAST = TILE - 1  # the last shared-memory slot

TYPES = (int64[:], int64[:], int64[:], int64, int64)


def kernel(a: np.ndarray, b: np.ndarray, out: np.ndarray, m: int, p: int) -> None:
    buf = cuda.shared.array(TILE, int64)
    splits = cuda.shared.array(2, int64)
    vals = cuda.local.array(ITEMS, int64)

    tid = cuda.threadIdx.x
    total = m + p
    k0 = int64(cuda.blockIdx.x) * TILE
    tile_len = min(total - k0, TILE)  # outputs of this tile (none past the end)

    # 1. Split of diagonal k: the first i in [max(0, k - p), min(k, m)) with
    #    a[i] > b[k - 1 - i], or min(k, m) if there is none.
    if tid == 0 or tid == 32:
        k = k0 if tid == 0 else k0 + tile_len
        lo = max(k - p, 0)
        hi = min(k, m)
        if m > 0 and p > 0:
            while lo < hi:
                mid = (lo + hi) >> 1
                if a[min(max(mid, 0), m - 1)] <= b[min(max(k - 1 - mid, 0), p - 1)]:
                    lo = mid + 1
                else:
                    hi = mid
        splits[(tid >> 5) & 1] = lo
    cuda.syncthreads()
    a_start = splits[0]
    b_start = k0 - a_start
    na = min(max(splits[1] - a_start, 0), max(tile_len, 0))
    nb = tile_len - na

    # 2. Tile inputs: a slice in buf[0:na], b slice in buf[na:tile_len].
    x = tid
    while x < tile_len:
        v = int64(0)
        if x < na:
            i = a_start + x
            if i >= 0 and i < m:
                v = a[i]
        else:
            i = b_start + x - na
            if i >= 0 and i < p:
                v = b[i]
        buf[x] = v
        x += BLOCK
    cuda.syncthreads()

    # 3. Split of this thread's diagonal, then a sequential merge of ITEMS outputs.
    d = min(tid * ITEMS, tile_len)
    lo = max(d - nb, 0)
    hi = min(d, na)
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf[min(max(mid, 0), LAST)] <= buf[min(max(na + d - 1 - mid, 0), LAST)]:
            lo = mid + 1
        else:
            hi = mid
    ai = lo
    bi = d - lo
    j = 0
    while j < ITEMS:
        va = buf[min(max(ai, 0), LAST)]
        vb = buf[min(max(na + bi, 0), LAST)]
        if ai < na and (bi >= nb or va <= vb):
            vals[j] = va
            ai += 1
        elif bi < nb:
            vals[j] = vb
            bi += 1
        j += 1
    cuda.syncthreads()

    # 4. Transpose through shared memory and store coalesced.
    j = 0
    while j < ITEMS:
        x = tid * ITEMS + j
        if x < tile_len:
            buf[x] = vals[j]
        j += 1
    cuda.syncthreads()
    x = tid
    while x < tile_len:
        k = k0 + x
        if k < total:
            out[k] = buf[x]
        x += BLOCK


_kernel = cuda.jit(kernel)


def run(a: DeviceNDArray, b: DeviceNDArray, out: DeviceNDArray, m: int, p: int) -> None:
    total = m + p
    if total == 0:
        return
    grid = (total + TILE - 1) // TILE
    _kernel[grid, BLOCK](a, b, out, m, p)
