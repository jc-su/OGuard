"""bloom_probe: match[i] = 1 when both hashed bits of keys[i] are set in the filter."""

import numba
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256


def kernel(
    keys: DeviceNDArray,
    bits: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
    words: int,
) -> None:
    i = cuda.grid(1)
    if i < n:
        key = numba.uint64(keys[i])
        mask = numba.uint64(32 * words - 1)
        h1 = (key * numba.uint64(0x9E3779B97F4A7C15)) >> numba.uint64(32)
        h2 = (key * numba.uint64(0xC2B2AE3D27D4EB4F)) >> numba.uint64(32)
        p1 = h1 & mask
        p2 = h2 & mask
        w1 = bits[p1 >> numba.uint64(5)] >> numba.uint32(p1 & numba.uint64(31))
        w2 = bits[p2 >> numba.uint64(5)] >> numba.uint32(p2 & numba.uint64(31))
        if (w1 & w2) & numba.uint32(1):
            match[i] = 1
        else:
            match[i] = 0


TYPES = (
    types.int64[:],
    types.uint32[:],
    types.int8[:],
    types.int64,
    types.int64,
)

_jitted = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    bits: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
    words: int,
) -> None:
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _jitted[blocks, THREADS](keys, bits, match, n, words)
