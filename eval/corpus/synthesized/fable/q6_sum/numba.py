"""q6_sum: TPC-H Q6 filtered sum of price * discount.

Each thread accumulates a private sum over a grid-stride loop, evaluating the
cheap date predicate first so the other columns are read only for candidate
rows. The block then reduces with warp shuffles and shared memory, and one
atomicAdd per block accumulates into result.

result must be zero on entry.
"""

import numpy as np
from numba import cuda, float64, int32, int64

FULL_MASK = 0xFFFFFFFF
THREADS = 256
MAX_BLOCKS = 2048


def kernel(shipdate, discount, quantity, price, date_lo, date_hi, disc_lo, disc_hi, qty_max, result, n):
    acc = 0.0
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        d = shipdate[i]
        if d >= date_lo and d < date_hi:
            disc = discount[i]
            qty = quantity[i]
            if disc >= disc_lo and disc <= disc_hi and qty < qty_max:
                acc += price[i] * disc
        i += stride

    acc += cuda.shfl_xor_sync(FULL_MASK, acc, 16)
    acc += cuda.shfl_xor_sync(FULL_MASK, acc, 8)
    acc += cuda.shfl_xor_sync(FULL_MASK, acc, 4)
    acc += cuda.shfl_xor_sync(FULL_MASK, acc, 2)
    acc += cuda.shfl_xor_sync(FULL_MASK, acc, 1)

    warp_sums = cuda.shared.array(32, float64)
    lane = cuda.laneid
    warp = cuda.threadIdx.x >> 5
    if lane == 0:
        warp_sums[warp] = acc
    cuda.syncthreads()
    if warp == 0:
        acc = 0.0
        if lane < (cuda.blockDim.x >> 5):
            acc = warp_sums[lane]
        acc += cuda.shfl_xor_sync(FULL_MASK, acc, 16)
        acc += cuda.shfl_xor_sync(FULL_MASK, acc, 8)
        acc += cuda.shfl_xor_sync(FULL_MASK, acc, 4)
        acc += cuda.shfl_xor_sync(FULL_MASK, acc, 2)
        acc += cuda.shfl_xor_sync(FULL_MASK, acc, 1)
        if lane == 0:
            cuda.atomic.add(result, 0, acc)


TYPES = (int32[:], float64[:], float64[:], float64[:], int32, int32, float64, float64, float64, float64[:], int64)

_kernel = cuda.jit(kernel)


def run(shipdate, discount, quantity, price, date_lo, date_hi, disc_lo, disc_hi, qty_max, result, n):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](
        shipdate,
        discount,
        quantity,
        price,
        np.int32(date_lo),
        np.int32(date_hi),
        float(disc_lo),
        float(disc_hi),
        float(qty_max),
        result,
        n,
    )
