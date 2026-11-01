"""bloom_build: set two hashed bits per key in a Bloom filter of 32 * words bits."""

import numba
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256


def kernel(
    keys: DeviceNDArray,
    bits: DeviceNDArray,
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
        word1 = p1 >> numba.uint64(5)
        word2 = p2 >> numba.uint64(5)
        bit1 = numba.uint32(1) << numba.uint32(p1 & numba.uint64(31))
        bit2 = numba.uint32(1) << numba.uint32(p2 & numba.uint64(31))
        # numba-cuda's stubs type cuda.atomic.or_ as a Stub class, not as a call.
        cuda.atomic.or_(bits, word1, bit1)  # ty: ignore[too-many-positional-arguments]
        cuda.atomic.or_(bits, word2, bit2)  # ty: ignore[too-many-positional-arguments]


TYPES = (
    types.int64[:],
    types.uint32[:],
    types.int64,
    types.int64,
)

_jitted = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    bits: DeviceNDArray,
    n: int,
    words: int,
) -> None:
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _jitted[blocks, THREADS](keys, bits, n, words)
