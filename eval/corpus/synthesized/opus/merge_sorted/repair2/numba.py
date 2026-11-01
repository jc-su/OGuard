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

Bounds: every access is proved by the code at the access alone, without a
minimum, a maximum or a loop invariant. A global access runs only under the
two comparisons 0 <= index and index < length of the very index it uses (m,
p, or m + p for out). The shared-memory tile has SLOTS = 2048 >= TILE + 1
entries and every index into it is masked with SLOTS - 1; a loop over the
tile takes its position through that mask too, or advances its slot with it
(slot = (slot + 1) & mask), so that the compiler forms each address from the
tested index instead of a pointer that steps with the loop. The ITEMS merged
outputs of a thread live in a per-thread array of VALUE_SLOTS = 8 entries
whose slot is advanced with the mask VALUE_SLOTS - 1; with the fixed loop
count the compiler keeps them in registers. None of these tests changes a
result for input that meets the contract: inside the searches a probe of a
lies in [0, m) and a probe of b in [0, p), and every tile index lies in
[0, TILE].
"""

import numpy as np
from numba import cuda, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
ITEMS = 7
TILE = BLOCK * ITEMS
SLOTS = 2048  # a power of two >= TILE + 1: the size of a shared-memory tile
MASK = SLOTS - 1
VALUE_SLOTS = 8  # a power of two >= ITEMS: the per-thread outputs
VALUE_MASK = VALUE_SLOTS - 1

TYPES = (int64[:], int64[:], int64[:], int64, int64)


def kernel(a: np.ndarray, b: np.ndarray, out: np.ndarray, m: int, p: int) -> None:
    buf = cuda.shared.array(SLOTS, int64)
    splits = cuda.shared.array(2, int64)
    vals = cuda.local.array(VALUE_SLOTS, int64)

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
        while lo < hi:
            mid = (lo + hi) >> 1
            ib = k - 1 - mid
            va = int64(0)
            vb = int64(0)
            if mid >= 0 and mid < m:
                va = a[mid]
            if ib >= 0 and ib < p:
                vb = b[ib]
            if va <= vb:
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
        s = x & MASK  # x itself
        v = int64(0)
        if s < na:
            i = a_start + s
            if i >= 0 and i < m:
                v = a[i]
        else:
            i = b_start + s - na
            if i >= 0 and i < p:
                v = b[i]
        buf[s] = v
        x += BLOCK
    cuda.syncthreads()

    # 3. Split of this thread's diagonal, then a sequential merge of ITEMS outputs.
    d = min(tid * ITEMS, tile_len)
    lo = max(d - nb, 0)
    hi = min(d, na)
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf[mid & MASK] <= buf[(na + d - 1 - mid) & MASK]:
            lo = mid + 1
        else:
            hi = mid
    ai = lo
    bi = d - lo
    v = 0  # the slot of output j in vals, advanced with the mask
    j = 0
    while j < ITEMS:
        va = buf[ai & MASK]
        vb = buf[(na + bi) & MASK]
        if ai < na and (bi >= nb or va <= vb):
            vals[v] = va
            ai += 1
        else:
            vals[v] = vb
            bi += 1
        v = (v + 1) & VALUE_MASK
        j += 1
    cuda.syncthreads()

    # 4. The merged outputs go back to the tile, then out with coalesced stores.
    v = 0
    s = d & MASK  # d + j, advanced with the mask
    j = 0
    while j < ITEMS:
        if d + j < tile_len:
            buf[s] = vals[v]
        v = (v + 1) & VALUE_MASK
        s = (s + 1) & MASK
        j += 1
    cuda.syncthreads()
    x = tid
    while x < tile_len:
        s = x & MASK
        k = k0 + s
        if k >= 0 and k < total:
            out[k] = buf[s]
        x += BLOCK


_kernel = cuda.jit(kernel)


def run(a: DeviceNDArray, b: DeviceNDArray, out: DeviceNDArray, m: int, p: int) -> None:
    total = m + p
    if total == 0:
        return
    grid = (total + TILE - 1) // TILE
    _kernel[grid, BLOCK](a, b, out, m, p)
