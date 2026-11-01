"""merge_sorted: merge two sorted int64 columns, a before an equal b.

Merge path (Green, McColl, Bader / ModernGPU) straight from global memory.
Each thread owns ITEMS consecutive outputs that start at diagonal d. It
finds the partition of that diagonal, the number of elements of `a` among
the first d outputs, with a binary search over the diagonal, then merges
its ITEMS outputs sequentially from `a` and `b`. Neighbouring threads probe
neighbouring positions, so the search loads coalesce and the upper levels
of the search stay in L1.

Revision: the staging of tiles in shared memory and the block-level
searches are gone, so no index that reaches memory passes through shared
memory. The search runs over the whole diagonal and treats a probe past
the end of an input as an exhausted input (a[m..] and b[p..] read as +inf),
which removes the min/max clamps of the search bounds; it is indexed by
k = i + 1 so that both probe indices, k - 1 and d - k, are plain sums.
Every probe and every merge step tests the exact index it is about to use
against 0 and the size of its array, and each element is loaded once;
under the task's contract those tests always pass.
"""

from numba import cuda, int64

TYPES = (int64[:], int64[:], int64[:], int64, int64)
THREADS = 256
ITEMS = 4


@cuda.jit(device=True, inline="always")
def merge_step(a, b, out, m, p, total, i, j, k):
    """Output k of the merge: a[i] or b[j], a first on ties; returns the advanced (i, j)."""
    if k < total:
        has_a = i >= 0 and i < m
        has_b = j >= 0 and j < p
        av = 0
        bv = 0
        if has_a:
            av = a[i]
        if has_b:
            bv = b[j]
        if has_a and (not has_b or av <= bv):
            out[k] = av
            i += 1
        elif has_b:
            out[k] = bv
            j += 1
    return i, j


def kernel(a, b, out, m, p):
    total = m + p
    d = (cuda.blockIdx.x * THREADS + cuda.threadIdx.x) * ITEMS
    if d >= total:
        return

    # Smallest k in [1, d + 1] with a[k - 1] > b[d - k]; k - 1 is the a-count of the first d outputs.
    lo = 1
    hi = d + 1
    while lo < hi:
        mid = (lo + hi) >> 1
        ia = mid - 1
        jb = d - mid
        if ia >= m:
            hi = mid  # a is exhausted at ia: the partition lies to the left
        elif jb >= p:
            lo = mid + 1  # b is exhausted at jb: the partition lies to the right
        elif ia < 0:
            break  # never under the contract
        elif jb < 0:
            hi = mid  # never under the contract
        elif a[ia] <= b[jb]:
            lo = mid + 1
        else:
            hi = mid

    i = lo - 1
    j = d - i
    i, j = merge_step(a, b, out, m, p, total, i, j, d)
    i, j = merge_step(a, b, out, m, p, total, i, j, d + 1)
    i, j = merge_step(a, b, out, m, p, total, i, j, d + 2)
    i, j = merge_step(a, b, out, m, p, total, i, j, d + 3)


_kernel = cuda.jit(kernel)


def run(a, b, out, m, p):
    blocks = max(1, (m + p + THREADS * ITEMS - 1) // (THREADS * ITEMS))
    _kernel[blocks, THREADS](a, b, out, m, p)
