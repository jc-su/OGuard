"""Numba CUDA kernel for filter_select.

Writes the row ids i where quantity[i] < threshold into sel, in any
order, and the number of matches into count.

count is an output buffer, so its value on entry is not part of the
task's contract: the checker must prove every store safe for whatever
count already holds. The atomic counter can therefore return a
position outside [0, n) if count did not start at zero, so the store
into sel is guarded by an explicit two-sided bounds check before it
runs. Under the documented precondition that the caller zeroes count
before each launch, the guard never rejects a valid position, because
at most n rows can match and each match gets a distinct position in
[0, n).
"""

from numba import cuda, float64, int32, int64

TYPES = (float64[:], int32[:], int32[:], float64, int64)

BLOCK = 256


def kernel(quantity, sel, count, threshold, n):
    i = cuda.grid(1)
    if i < n and quantity[i] < threshold:
        pos = cuda.atomic.add(count, 0, 1)
        if pos >= 0 and pos < n:
            sel[pos] = i


def run(quantity, sel, count, threshold, n):
    """Launch kernel on device arrays. count must already be zeroed."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](quantity, sel, count, threshold, n)
