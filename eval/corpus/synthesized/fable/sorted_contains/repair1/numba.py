"""sorted_contains: membership of each key in a sorted, distinct member list.

One thread per key: lower-bound binary search in `members`, then an equality
test. The upper levels of the search tree are shared by every thread and
stay in L1/L2; s = 0 gives no match.

Revision: every probe of `members` is guarded by an explicit range test on
its index. The search keeps 0 <= lo < hi <= s, so the tests always pass; they
state the bound of each access for the memory checker, which does not carry
that loop invariant.
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
        if mid < 0 or mid >= s:
            break
        if members[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    found = 0
    if lo >= 0 and lo < s:
        if members[lo] == key:
            found = 1
    match[i] = found


_kernel = cuda.jit(kernel)


def run(keys, members, match, n, s):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](keys, members, match, n, s)
