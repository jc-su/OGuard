"""string_hash: out[i] = 32-bit FNV-1a hash of the bytes of string i.

FNV-1a is sequential within a string, so each thread hashes one string.
Neighboring threads hash neighboring strings, so their byte loads fall in the same
cache lines.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619

TYPES = (
    types.int32[:],  # offsets
    types.uint8[:],  # chars
    types.uint32[:],  # out
    types.int64,  # n
)


def kernel(offsets, chars, out, n):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        h = types.uint32(FNV_OFFSET)
        for p in range(offsets[i], offsets[i + 1]):
            h = types.uint32((h ^ chars[p]) * types.uint32(FNV_PRIME))
        out[i] = h
        i += stride


_kernel = cuda.jit(kernel)


def run(offsets, chars, out, n):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](offsets, chars, out, n)
