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
hold by induction across the loop and by construction of the split.
A min/max clamp applied to an index right before the read expresses the
same invariant as an algebraic fact about the clamped value, but a
compiler is free to prove that fact redundant (given the loop's actual
bracket) and fold the clamp away, leaving the read gated by nothing a
static checker can use. Every read of `a` or `b` below is instead gated
by an `if` whose condition is the exact bound comparison on the index
used in that read, with the read itself lexically inside the branch, so
the guard cannot be separated from the access it protects.
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
                bj = j
                if ai >= 0 and ai < m and bj >= 0 and bj < p:
                    step_down = a[ai] > b[bj]
            if step_down:
                hi = i - 1
            else:
                lo = i
        i = lo
        j = k - i

        take_a = False
        if i >= 0 and i < m:
            if j >= p:
                take_a = True
            elif j >= 0 and j < p:
                take_a = a[i] <= b[j]

        if take_a:
            if i >= 0 and i < m:
                out[k] = a[i]
        else:
            if j >= 0 and j < p:
                out[k] = b[j]


def run(a, b, out, m, p):
    total = m + p
    if total <= 0:
        return
    blocks = (total + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](a, b, out, m, p)
