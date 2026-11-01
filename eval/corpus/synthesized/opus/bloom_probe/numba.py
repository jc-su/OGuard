"""bloom_probe: match[i] = 1 when both hash bits of keys[i] are set in the filter.

The bit positions are those of bloom_build: the high 32 bits of the key times
one of two odd 64-bit constants, in wrapping unsigned arithmetic, modulo the
32 * words bits of the filter. words is a power of two, so the remainder is a
mask. The second word is read only when the first bit is set.
"""

import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
MULTIPLIER_1 = np.uint64(0x9E3779B97F4A7C15)
MULTIPLIER_2 = np.uint64(0xC2B2AE3D27D4EB4F)


def kernel(
    keys: DeviceNDArray,
    bits: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
    words: int,
) -> None:
    i = cuda.grid(1)
    if i < n:
        mask = np.uint64(words) * np.uint64(32) - np.uint64(1)
        key = np.uint64(keys[i])
        p1 = ((key * MULTIPLIER_1) >> np.uint64(32)) & mask
        p2 = ((key * MULTIPLIER_2) >> np.uint64(32)) & mask
        word1 = bits[p1 >> np.uint64(5)]
        if (word1 >> np.uint32(p1 & np.uint64(31))) & np.uint32(1):
            word2 = bits[p2 >> np.uint64(5)]
            match[i] = (word2 >> np.uint32(p2 & np.uint64(31))) & np.uint32(1)
        else:
            match[i] = 0


TYPES = (
    types.int64[:],
    types.uint32[:],
    types.int8[:],
    types.int64,
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    bits: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
    words: int,
) -> None:
    blocks = (n + THREADS - 1) // THREADS
    if blocks > 0:
        _kernel[blocks, THREADS](keys, bits, match, n, words)
