"""starts_with: match[i] = 1 when string i begins with the k-byte pattern prefix.

One thread per string: a string shorter than k never matches; otherwise the
first k bytes are compared with the pattern and the loop stops at the first
mismatch. The pattern is tiny and uniform across threads, so its loads are
served from L1 as broadcasts. The byte loop counts j in int64 against k
itself, so j < k bounds prefix[j] by the pattern length and, with
offsets[i + 1] - begin >= k, bounds chars[begin + j] by the end of string i.
"""

from numba import cuda, int8, int32, int64, uint8

THREADS = 256
MAX_BLOCKS = 2048


def kernel(offsets, chars, prefix, match, n, k):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        begin = offsets[i]
        ok = offsets[i + 1] - begin >= k
        j = 0
        while ok and j < k:
            ok = chars[begin + j] == prefix[j]
            j += 1
        match[i] = 1 if ok else 0
        i += stride


TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(offsets, chars, prefix, match, n, k):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](offsets, chars, prefix, match, n, k)
