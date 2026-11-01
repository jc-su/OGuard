"""merge_sorted: merge two ascending int64 arrays, a before b on ties (Numba CUDA).

A merge-path merge. Each block produces the tile out[k0:k1] of TILE outputs:

1. Split: find how many of the first k0 (and k1) outputs come from a, by a
   binary search along the cross diagonal (lane 0 of warp 0 for k0, lane 0
   of warp 1 for k1, in parallel). With ties the split takes a first
   (a[i] <= b[k - 1 - i]). The probes of a binary search close in on the
   split, near data the block reads next; a 256-ary search with one probe
   per thread measured 3x slower, as its first rounds read hundreds of
   scattered sectors per block.
2. Load the tile's slices of a and b into shared memory with coalesced reads.
3. Each thread finds the split of its own diagonal (ITEMS outputs apart) in
   shared memory and merges ITEMS outputs sequentially into registers.
4. The merged tile goes back through shared memory so the stores coalesce.

ITEMS is odd so that the strided shared-memory accesses of step 4 have no
bank conflicts.
"""

import numpy as np
from numba import cuda, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
ITEMS = 7
TILE = BLOCK * ITEMS

TYPES = (int64[:], int64[:], int64[:], int64, int64)


def kernel(a: np.ndarray, b: np.ndarray, out: np.ndarray, m: int, p: int) -> None:
    buf = cuda.shared.array(TILE, int64)
    splits = cuda.shared.array(2, int64)
    vals = cuda.local.array(ITEMS, int64)

    tid = cuda.threadIdx.x
    k0 = int64(cuda.blockIdx.x) * TILE
    k1 = min(k0 + TILE, m + p)

    # 1. Split of diagonal k: the first i in [max(0, k - p), min(k, m)) with
    #    a[i] > b[k - 1 - i], or min(k, m) if there is none.
    if tid == 0 or tid == 32:
        k = k0 if tid == 0 else k1
        lo = max(k - p, 0)
        hi = min(k, m)
        while lo < hi:
            mid = (lo + hi) >> 1
            if a[mid] <= b[k - 1 - mid]:
                lo = mid + 1
            else:
                hi = mid
        splits[tid >> 5] = lo
    cuda.syncthreads()
    lo0 = splits[0]
    lo1 = splits[1]
    a_start = lo0
    b_start = k0 - lo0
    na = lo1 - lo0
    nb = (k1 - lo1) - b_start
    tile_len = na + nb

    # 2. Tile inputs: a slice in buf[0:na], b slice in buf[na:tile_len].
    for x in range(tid, tile_len, BLOCK):
        if x < na:
            buf[x] = a[a_start + x]
        else:
            buf[x] = b[b_start + x - na]
    cuda.syncthreads()

    # 3. Split of this thread's diagonal, then a sequential merge of ITEMS outputs.
    d = min(tid * ITEMS, tile_len)
    lo = max(d - nb, 0)
    hi = min(d, na)
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf[mid] <= buf[na + d - 1 - mid]:
            lo = mid + 1
        else:
            hi = mid
    ai = lo
    bi = d - lo
    for j in range(ITEMS):
        if ai < na and (bi >= nb or buf[ai] <= buf[na + bi]):
            vals[j] = buf[ai]
            ai += 1
        elif bi < nb:
            vals[j] = buf[na + bi]
            bi += 1
    cuda.syncthreads()

    # 4. Transpose through shared memory and store coalesced.
    for j in range(ITEMS):
        x = tid * ITEMS + j
        if x < tile_len:
            buf[x] = vals[j]
    cuda.syncthreads()
    for x in range(tid, tile_len, BLOCK):
        out[k0 + x] = buf[x]


_kernel = cuda.jit(kernel)


def run(a: DeviceNDArray, b: DeviceNDArray, out: DeviceNDArray, m: int, p: int) -> None:
    total = m + p
    if total == 0:
        return
    grid = (total + TILE - 1) // TILE
    _kernel[grid, BLOCK](a, b, out, m, p)
