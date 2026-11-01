"""Numba CUDA kernel: sorted_contains.

match[i] = 1 if keys[i] is one of the sorted, distinct `members`, else 0.
One thread per key, each doing its own binary search over `members`
(length s): O(n log s) total work, fully parallel with no cross-thread
state.

The loop invariant `0 <= lo <= hi <= s` (so `mid` always lands inside
`members`) holds by induction on lo/hi across iterations. A min/max
clamp applied to `mid` right before the read expresses the same
invariant algebraically, but a compiler is free to prove that fact
redundant (given the loop's own bracket) and fold the clamp away,
leaving the read gated by nothing a static checker can use. The read
below is instead gated by an `if` whose condition is the exact bound
comparison on `mid`, with the read itself lexically inside the branch,
so the guard cannot be separated from the access it protects.
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
            advance = False
            if mid >= 0 and mid < s:
                advance = members[mid] < key
            if advance:
                lo = mid + 1
            else:
                hi = mid

        found = False
        if lo >= 0 and lo < s:
            found = members[lo] == key
        match[i] = int8(1) if found else int8(0)


def run(keys, members, match, n, s):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](keys, members, match, n, s)
