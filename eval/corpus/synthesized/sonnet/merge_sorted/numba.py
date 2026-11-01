"""Numba CUDA kernel: merge_sorted.

Merges two ascending-sorted arrays `a` (length m) and `b` (length p)
into `out` (length m + p), ascending, with an element of `a` placed
before an equal element of `b`.

Uses the "merge path" (co-rank) algorithm: for each output position k,
a binary search over the diagonal finds the split (i, j) with i + j = k
such that the first k output elements are exactly a[0:i] merged with
b[0:j]. Each of the m + p output positions is then produced by an
independent thread with no synchronization, in O((m + p) log(min(m, p)))
total work -- the standard way to parallelize a stable merge on a GPU.
"""

from numba import cuda, int64

TYPES = (int64[:], int64[:], int64[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(a, b, out, m, p):
    k = cuda.grid(1)
    total = m + p
    if k < total:
        lo = max(int64(0), k - p)
        hi = min(k, m)
        while lo < hi:
            i = (lo + hi + 1) // 2
            j = k - i
            if i > 0 and j < p and a[i - 1] > b[j]:
                hi = i - 1
            else:
                lo = i
        i = lo
        j = k - i
        if i < m and (j >= p or a[i] <= b[j]):
            out[k] = a[i]
        else:
            out[k] = b[j]


def run(a, b, out, m, p):
    total = m + p
    if total <= 0:
        return
    blocks = (total + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](a, b, out, m, p)
