"""starts_with: match[i] = 1 when string i begins with the k-byte pattern prefix.

One thread per string: a string shorter than k never matches; otherwise the
first k bytes are compared with the pattern and the loop stops at the first
mismatch. The pattern is tiny and uniform across threads, so its loads are
served from L1 as broadcasts. Byte positions are unsigned so Numba skips the
negative-index wraparound check on every chars[] load.
"""

from numba import cuda, int8, int32, int64, uint8, uint32

THREADS = 256
MAX_BLOCKS = 2048


def kernel(offsets, chars, prefix, match, n, k):
    kk = uint32(k)
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        begin = uint32(offsets[i])
        ok = uint32(offsets[i + 1]) - begin >= kk
        j = uint32(0)
        while ok and j < kk:
            ok = chars[begin + j] == prefix[j]
            j += uint32(1)
        match[i] = 1 if ok else 0
        i += stride


TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(offsets, chars, prefix, match, n, k):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](offsets, chars, prefix, match, n, k)
