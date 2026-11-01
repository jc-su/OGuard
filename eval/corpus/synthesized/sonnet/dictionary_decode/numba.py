"""Numba CUDA kernel: dictionary_decode.

out[i] = dictionary[codes[i]] for i in [0, n). Purely elementwise and
memory-bound: one thread per row, a coalesced read of `codes[i]`, a
gather from `dictionary`, and a coalesced write to `out[i]`.
"""

from numba import cuda, float64, int32, int64

TYPES = (int32[:], float64[:], float64[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(codes, dictionary, out, n, d):
    i = cuda.grid(1)
    if i < n:
        out[i] = dictionary[codes[i]]


def run(codes, dictionary, out, n, d):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](codes, dictionary, out, n, d)
