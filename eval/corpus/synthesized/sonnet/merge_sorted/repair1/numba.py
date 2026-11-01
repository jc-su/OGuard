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

The invariants that keep every index below inside its buffer (the
binary search's `lo/hi` bracket, and the co-rank split `i + j = k`)
hold by induction across the loop and by construction of the split, but
a static bound checker need not perform either piece of reasoning:
every index used to read `a` or `b` is clamped with local min/max
guards immediately before the read, so each bound follows from those
comparisons alone.
"""

from numba import cuda, int64

TYPES = (int64[:], int64[:], int64[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(a, b, out, m, p):
    # Compute the output position in explicit 64-bit arithmetic:
    # cuda.grid(1) combines blockIdx/blockDim/threadIdx in 32-bit
    # registers, and for an adversarial (oversized) grid that product can
    # overflow 32 bits.
    k = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    total = m + p
    if k >= 0 and k < total:
        lo = max(int64(0), k - p)
        hi = min(k, m)
        while lo < hi:
            i = (lo + hi + 1) // 2
            j = k - i
            step_down = False
            if i > 0 and j < p:
                ai = i - 1
                if ai < 0:
                    ai = 0
                if ai >= m:
                    ai = m - 1
                bj = j
                if bj < 0:
                    bj = 0
                if bj >= p:
                    bj = p - 1
                step_down = a[ai] > b[bj]
            if step_down:
                hi = i - 1
            else:
                lo = i
        i = lo
        j = k - i

        take_a = False
        if i < m:
            if j >= p:
                take_a = True
            else:
                ai = i
                if ai < 0:
                    ai = 0
                if ai >= m:
                    ai = m - 1
                bj = j
                if bj < 0:
                    bj = 0
                if bj >= p:
                    bj = p - 1
                take_a = a[ai] <= b[bj]

        if take_a:
            ai = i
            if ai < 0:
                ai = 0
            if ai >= m:
                ai = m - 1
            out[k] = a[ai]
        else:
            bj = j
            if bj < 0:
                bj = 0
            if bj >= p:
                bj = p - 1
            out[k] = b[bj]


def run(a, b, out, m, p):
    total = m + p
    if total <= 0:
        return
    blocks = (total + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](a, b, out, m, p)
