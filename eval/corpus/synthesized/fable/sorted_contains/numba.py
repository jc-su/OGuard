"""sorted_contains: membership of each key in a sorted, distinct member list.

One thread per key: lower-bound binary search in `members`, then an equality
test. The upper levels of the search tree are shared by every thread and
stay in L1/L2; s = 0 gives no match.
"""

from numba import cuda, int8, int64

TYPES = (int64[:], int64[:], int8[:], int64, int64)
THREADS = 256


def kernel(keys, members, match, n, s):
    i = cuda.grid(1)
    if i >= n:
        return
    key = keys[i]
    lo = 0
    hi = s
    while lo < hi:
        mid = (lo + hi) >> 1
        if members[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    found = 0
    if lo < s and members[lo] == key:
        found = 1
    match[i] = found


_kernel = cuda.jit(kernel)


def run(keys, members, match, n, s):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](keys, members, match, n, s)
