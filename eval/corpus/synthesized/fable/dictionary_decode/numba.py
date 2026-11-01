"""dictionary_decode: gather dictionary values through int32 codes.

One thread per row: out[i] = dictionary[codes[i]]. The code loads and the
output stores are coalesced; the dictionary gather is served by L1/L2.
"""

from numba import cuda, float64, int32, int64

TYPES = (int32[:], float64[:], float64[:], int64, int64)
THREADS = 256


def kernel(codes, dictionary, out, n, d):
    i = cuda.grid(1)
    if i < n:
        out[i] = dictionary[codes[i]]


_kernel = cuda.jit(kernel)


def run(codes, dictionary, out, n, d):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](codes, dictionary, out, n, d)
