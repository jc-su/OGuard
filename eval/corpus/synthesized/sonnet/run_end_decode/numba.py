"""Numba CUDA kernel: run_end_decode.

Decodes an Arrow run-end encoded column: out[i] = run_values[j] where run
j is the smallest run whose end exceeds row i, i.e. j = upper_bound(i) in
run_ends. One thread per output row, each doing its own binary search
over run_ends (length r); this is O(n log r) work but embarrassingly
parallel and memory-light (no shared state between threads), and r is
small enough in practice to stay cache-resident across the grid.
"""

from numba import cuda, float64, int32, int64

TYPES = (float64[:], int32[:], float64[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(run_values, run_ends, out, r, n):
    i = cuda.grid(1)
    if i < n:
        lo = int64(0)
        hi = int64(r)
        while lo < hi:
            mid = (lo + hi) >> 1
            if run_ends[mid] <= i:
                lo = mid + 1
            else:
                hi = mid
        out[i] = run_values[lo]


def run(run_values, run_ends, out, r, n):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](run_values, run_ends, out, r, n)
