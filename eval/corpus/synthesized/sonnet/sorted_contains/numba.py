"""Numba CUDA kernel: sorted_contains.

match[i] = 1 if keys[i] is one of the sorted, distinct `members`, else 0.
One thread per key, each doing its own binary search over `members`
(length s): O(n log s) total work, fully parallel with no cross-thread
state.
"""

from numba import cuda, int8, int64

TYPES = (int64[:], int64[:], int8[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(keys, members, match, n, s):
    i = cuda.grid(1)
    if i < n:
        key = keys[i]
        lo = int64(0)
        hi = int64(s)
        while lo < hi:
            mid = (lo + hi) >> 1
            if members[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        found = (lo < s) and (members[lo] == key)
        match[i] = int8(1) if found else int8(0)


def run(keys, members, match, n, s):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](keys, members, match, n, s)
