"""run_end_decode: decode an Arrow run-end encoded float64 column (Numba CUDA).

Each block decodes a tile of TILE consecutive rows [t0, t0 + tile_len):

1. Find the first run of the tile, the first j with run_ends[j] > t0, by a
   256-ary search: in each round every thread tests one splitter of the
   remaining range and syncthreads_count gives the sub-range that holds the
   answer. A range of r runs takes ceil(log256(r)) rounds of one load per
   thread, instead of log2(r) dependent loads. The first rounds probe the same
   positions in every block, so they hit L2.
2. Stage the ends of the runs that cover the tile in shared memory, one
   coalesced chunk of BLOCK runs at a time, until a chunk holds the run that
   reaches the tile end. The tile's TILE rows meet at most TILE runs.
3. Each thread decodes ITEMS consecutive rows: a binary search over the
   staged ends finds the run of its first row (row t0 + x lies in one of the
   first x + 1 staged runs), and each next row is in the same run or the next
   one, which one comparison decides. The values go through shared memory so
   that the stores coalesce. ITEMS is odd so that the strided shared-memory
   accesses have no bank conflicts.

Bounds: every access is proved by the code at the access alone, without a
minimum, a maximum or a loop invariant. A global access runs only under the
two comparisons 0 <= index and index < length of the very index it uses (r
for run_ends and run_values, n for out). The shared arrays have SLOTS = 2048
>= TILE entries and every index into them is masked with SLOTS - 1. A loop
over the tile advances its slot with that mask (slot = (slot + step) & mask),
so that the compiler forms each address from the masked or tested index
instead of a pointer that steps with the loop. None of these tests changes a
result for input that meets the contract.
"""

import numpy as np
from numba import cuda, float64, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
ITEMS = 7
TILE = BLOCK * ITEMS
SLOTS = 2048  # a power of two >= TILE: the size of a shared-memory tile
MASK = SLOTS - 1
INT32_MAX = 2147483647

TYPES = (float64[:], int32[:], float64[:], int64, int64)


def kernel(run_values: np.ndarray, run_ends: np.ndarray, out: np.ndarray, r: int, n: int) -> None:
    ends = cuda.shared.array(SLOTS, int32)
    stage = cuda.shared.array(SLOTS, float64)

    tid = cuda.threadIdx.x
    t0 = int64(cuda.blockIdx.x) * TILE
    tile_len = min(n - t0, TILE)  # rows of the tile (none past the end of the column)
    t1 = t0 + tile_len

    # 1. First j in [0, r - 1] with run_ends[j] > t0; run_ends[r - 1] = n > t0.
    #    Invariant: that j lies in [below + 1, hi], so run_ends[below] <= t0.
    first_run = int64(0)
    if r > 1:
        below = int64(-1)
        hi = r - 1
        while below + 1 < hi:
            step = (hi - below + BLOCK - 2) // BLOCK  # ceil((hi - below - 1) / BLOCK)
            probe = below + (tid + 1) * step
            pred = 0
            if probe < hi and probe >= 0 and probe < r:
                if run_ends[probe] <= t0:
                    pred = 1
            below += cuda.syncthreads_count(pred) * step
            hi = min(below + step, hi)
        first_run = below + 1

    # 2. Stage run ends until a chunk holds the run that reaches t1.
    loaded = 0
    slot = tid & MASK  # loaded + tid
    while loaded + BLOCK <= TILE:
        j = first_run + slot
        e = INT32_MAX
        if j >= 0 and j < r:
            e = run_ends[j]
        ends[slot] = e
        loaded += BLOCK
        slot = (slot + BLOCK) & MASK
        done = 0
        if e >= t1:
            done = 1
        if cuda.syncthreads_count(done) > 0:
            break

    # 3. Decode rows t0 + [x0, x0 + ITEMS): run of row t0 + x = first staged a with ends[a] > row.
    x0 = tid * ITEMS
    if x0 < tile_len:
        row = t0 + x0
        a = 0
        if ends[0] <= row:
            a = 1
            b = min(loaded, x0 + 1)
            while a < b:
                mid = (a + b) >> 1
                if ends[mid & MASK] <= row:
                    a = mid + 1
                else:
                    b = mid
        x = x0 & MASK  # x0 + j, advanced with the mask
        j = 0
        while j < ITEMS:
            if x0 + j < tile_len:
                if ends[a & MASK] <= row + j:
                    a += 1
                v = first_run + a
                if v >= 0 and v < r:
                    stage[x] = run_values[v]
            x = (x + 1) & MASK
            j += 1
    cuda.syncthreads()
    x = tid
    while x < tile_len:
        s = x & MASK  # x itself
        g = t0 + s
        if g >= 0 and g < n:
            out[g] = stage[s]
        x += BLOCK


_kernel = cuda.jit(kernel)


def run(
    run_values: DeviceNDArray, run_ends: DeviceNDArray, out: DeviceNDArray, r: int, n: int
) -> None:
    if n == 0:
        return
    grid = (n + TILE - 1) // TILE
    _kernel[grid, BLOCK](run_values, run_ends, out, r, n)
