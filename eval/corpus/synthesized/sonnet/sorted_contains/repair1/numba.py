"""Numba CUDA kernel: sorted_contains.

match[i] = 1 if keys[i] is one of the sorted, distinct `members`, else 0.
One thread per key, each doing its own binary search over `members`
(length s): O(n log s) total work, fully parallel with no cross-thread
state.

The loop invariant `0 <= lo <= hi <= s` (so `mid` always lands inside
`members`) holds by induction on lo/hi across iterations, but a static
bound checker need not perform that induction: `mid` is clamped to
[0, s) with two local min/max guards immediately before the read, so
the bound follows from those two comparisons alone.
"""

from numba import cuda, int8, int64

TYPES = (int64[:], int64[:], int8[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(keys, members, match, n, s):
    # Compute the row index in explicit 64-bit arithmetic: cuda.grid(1)
    # combines blockIdx/blockDim/threadIdx in 32-bit registers, and for an
    # adversarial (oversized) grid that product can overflow 32 bits.
    i = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i >= 0 and i < n:
        key = keys[i]
        lo = int64(0)
        hi = int64(s)
        while lo < hi:
            mid = (lo + hi) >> 1
            # Inside the loop, s >= 1 always: hi only ever shrinks from
            # s, and lo < hi (the loop condition) forces hi >= 1.
            safe_mid = mid
            if safe_mid < 0:
                safe_mid = 0
            if safe_mid >= s:
                safe_mid = s - 1
            if members[safe_mid] < key:
                lo = mid + 1
            else:
                hi = mid

        found = False
        if s > 0:
            safe_lo = lo
            if safe_lo < 0:
                safe_lo = 0
            if safe_lo >= s:
                safe_lo = s - 1
            found = (lo < s) and (members[safe_lo] == key)
        match[i] = int8(1) if found else int8(0)


def run(keys, members, match, n, s):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](keys, members, match, n, s)
