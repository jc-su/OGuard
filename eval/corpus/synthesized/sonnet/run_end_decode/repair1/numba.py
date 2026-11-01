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
that `run_ends[r - 1] == n`, but a static bound checker need not perform
either piece of reasoning: both `mid` and the final `lo` are clamped
with local min/max guards immediately before the read, so each bound
follows from those comparisons alone.
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
            # Inside the loop, r >= 1 always: hi only ever shrinks from
            # r, and lo < hi (the loop condition) forces hi >= 1.
            safe_mid = mid
            if safe_mid < 0:
                safe_mid = 0
            if safe_mid >= r:
                safe_mid = r - 1
            if run_ends[safe_mid] <= i:
                lo = mid + 1
            else:
                hi = mid

        safe_lo = lo
        if safe_lo < 0:
            safe_lo = 0
        if safe_lo >= r:
            safe_lo = r - 1 if r > 0 else 0
        out[i] = run_values[safe_lo] if r > 0 else 0.0


def run(run_values, run_ends, out, r, n):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](run_values, run_ends, out, r, n)
