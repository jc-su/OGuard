"""bloom_build: set two hash bits of each key in a Bloom filter of 32 * words bits.

Bit p of the filter is bit p % 32 of bits[p / 32]. The two bit positions of a
key are h1 % (32 * words) and h2 % (32 * words), where h1 and h2 are the high
32 bits of the key times one of two odd 64-bit constants, in wrapping unsigned
arithmetic. words is a power of two, so the remainder is a mask.
"""

import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
MULTIPLIERS = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F))


def kernel(
    keys: DeviceNDArray,
    bits: DeviceNDArray,
    n: int,
    words: int,
) -> None:
    i = cuda.grid(1)
    if i < n:
        mask = np.uint64(words) * np.uint64(32) - np.uint64(1)
        key = np.uint64(keys[i])
        for multiplier in MULTIPLIERS:
            p = ((key * multiplier) >> np.uint64(32)) & mask
            word = p >> np.uint64(5)
            bit = np.uint32(1) << np.uint32(p & np.uint64(31))
            # Numba compiles this call to an atomic OR. Its Python stub is a
            # class, and the type checker matches the arguments to its
            # constructor.
            cuda.atomic.or_(bits, word, bit)  # ty: ignore[too-many-positional-arguments]


TYPES = (
    types.int64[:],
    types.uint32[:],
    types.int64,
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    bits: DeviceNDArray,
    n: int,
    words: int,
) -> None:
    blocks = (n + THREADS - 1) // THREADS
    if blocks > 0:
        _kernel[blocks, THREADS](keys, bits, n, words)
