"""Numba CUDA kernel: run_end_decode.

Decodes an Arrow run-end encoded column: out[i] = run_values[j] where run
j is the smallest run whose end exceeds row i, i.e. j = upper_bound(i) in
run_ends. One thread per output row, each doing its own binary search
over run_ends (length r); this is O(n log r) work but embarrassingly
parallel and memory-light (no shared state between threads), and r is
small enough in practice to stay cache-resident across the grid.

The loop invariant `0 <= lo <= hi <= r` (so `mid` always lands inside
`run_ends`) holds by induction on lo/hi across iterations, and the final
`lo < r` (so `run_values[lo]` is in range) follows from the contract
that `run_ends[r - 1] == n`. A min/max clamp applied to `mid` (or to the
final `lo`) right before the read expresses the same invariant
algebraically, but a compiler is free to prove that fact redundant
(given the loop's own bracket) and fold the clamp away, leaving the
read gated by nothing a static checker can use. Every read below is
instead gated by an `if` whose condition is the exact bound comparison
on the index used in that read, with the read itself lexically inside
the branch, so the guard cannot be separated from the access it
protects.
"""

from numba import cuda, float64, int32, int64

TYPES = (float64[:], int32[:], float64[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(run_values, run_ends, out, r, n):
    # Compute the row index in explicit 64-bit arithmetic: cuda.grid(1)
    # combines blockIdx/blockDim/threadIdx in 32-bit registers, and for an
    # adversarial (oversized) grid that product can overflow 32 bits.
    i = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i >= 0 and i < n:
        lo = int64(0)
        hi = int64(r)
        while lo < hi:
            mid = (lo + hi) >> 1
            advance = False
            if mid >= 0 and mid < r:
                advance = run_ends[mid] <= i
            if advance:
                lo = mid + 1
            else:
                hi = mid

        if lo >= 0 and lo < r:
            out[i] = run_values[lo]
        else:
            out[i] = 0.0


def run(run_values, run_ends, out, r, n):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](run_values, run_ends, out, r, n)
