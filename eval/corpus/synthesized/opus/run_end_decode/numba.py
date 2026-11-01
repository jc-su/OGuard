"""run_end_decode: decode an Arrow run-end encoded float64 column (Numba CUDA).

Each block decodes a tile of TILE consecutive rows [t0, t1):

1. Find the first run of the tile, the first j with run_ends[j] > t0, by a
   256-ary search: in each round every thread tests one splitter of the
   remaining range and syncthreads_count gives the sub-range that holds the
   answer. A range of r runs takes ceil(log256(r)) rounds of one load per
   thread, instead of log2(r) dependent loads. The first rounds probe the same
   positions in every block, so they hit L2.
2. Stage the ends of the runs that cover the tile in shared memory, one
   coalesced chunk of BLOCK runs at a time, until a chunk holds the run that
   reaches t1. The tile's TILE rows meet at most TILE runs.
3. Each thread decodes ITEMS consecutive rows: a binary search over the
   staged ends finds the run of its first row (row t0 + x lies in one of the
   first x + 1 staged runs), and each next row is in the same run or the next
   one, which one comparison decides. The values go through shared memory so
   that the stores coalesce. ITEMS is odd so that the strided shared-memory
   accesses have no bank conflicts.
"""

import numpy as np
from numba import cuda, float64, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
ITEMS = 7
TILE = BLOCK * ITEMS
INT32_MAX = 2147483647

TYPES = (float64[:], int32[:], float64[:], int64, int64)


def kernel(run_values: np.ndarray, run_ends: np.ndarray, out: np.ndarray, r: int, n: int) -> None:
    ends = cuda.shared.array(TILE, int32)
    stage = cuda.shared.array(TILE, float64)

    tid = cuda.threadIdx.x
    t0 = int64(cuda.blockIdx.x) * TILE
    t1 = min(t0 + TILE, n)

    # 1. First j in [0, r - 1] with run_ends[j] > t0; run_ends[r - 1] = n > t0.
    lo = int64(0)
    hi = r - 1
    while lo < hi:
        step = (hi - lo + BLOCK - 1) // BLOCK
        probe = lo + (tid + 1) * step - 1
        pred = 0
        if probe < hi and run_ends[probe] <= t0:
            pred = 1
        lo += cuda.syncthreads_count(pred) * step
        hi = min(lo + step - 1, hi)
    first_run = lo

    # 2. Stage run ends until a chunk holds the run that reaches t1.
    loaded = 0
    for c in range(ITEMS):
        j = first_run + c * BLOCK + tid
        e = INT32_MAX
        if j < r:
            e = run_ends[j]
        ends[c * BLOCK + tid] = e
        loaded += BLOCK
        done = 0
        if e >= t1:
            done = 1
        if cuda.syncthreads_count(done) > 0:
            break

    # 3. Decode rows t0 + [x0, x0 + ITEMS): run of row t0 + x = first staged a with ends[a] > row.
    tile_len = t1 - t0
    x0 = tid * ITEMS
    if x0 < tile_len:
        row = t0 + x0
        a = 0
        if ends[0] <= row:
            a = 1
            b = min(loaded, x0 + 1)
            while a < b:
                mid = (a + b) >> 1
                if ends[mid] <= row:
                    a = mid + 1
                else:
                    b = mid
        for j in range(ITEMS):
            x = x0 + j
            if x < tile_len:
                if ends[a] <= t0 + x:
                    a += 1
                stage[x] = run_values[first_run + a]
    cuda.syncthreads()
    for x in range(tid, tile_len, BLOCK):
        out[t0 + x] = stage[x]


_kernel = cuda.jit(kernel)


def run(
    run_values: DeviceNDArray, run_ends: DeviceNDArray, out: DeviceNDArray, r: int, n: int
) -> None:
    if n == 0:
        return
    grid = (n + TILE - 1) // TILE
    _kernel[grid, BLOCK](run_values, run_ends, out, r, n)
