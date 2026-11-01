"""Numba CUDA kernel: string_contains.

match[i] = 1 if `pattern` occurs as a byte substring of string i, else 0.
The empty pattern matches every string.

One thread per row does a naive (Sunday/brute-force) substring scan of
its own string, reading `pattern` directly from global memory. The
original version staged `pattern` into a dynamically sized shared array
once per block, but `k` (pattern length) is unbounded by this task's
contract ("[0, 2^31)"), so no fixed-size shared buffer can be sized to
always hold it, and a dynamically sized `cuda.shared.array(0, ...)`
array has no size visible to any analysis of the kernel body alone (its
actual size is a launch-time argument), so no access into it can be
proved in bounds. Reading `pattern` straight from global memory drops
that one block-level reuse but keeps every access provable.
"""

from numba import cuda, int8, int32, int64, uint8

TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(offsets, chars, pattern, match, n, k):
    # Compute the row index in explicit 64-bit arithmetic: cuda.grid(1)
    # combines blockIdx/blockDim/threadIdx in 32-bit registers, and for an
    # adversarial (oversized) grid that product can overflow 32 bits.
    i = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i >= 0 and i < n:
        start = offsets[i]
        end = offsets[i + 1]
        length = end - start

        if k == 0:
            match[i] = 1
        elif k > length:
            match[i] = 0
        else:
            found = False
            last_start = length - k
            s = 0
            while s <= last_start and not found:
                ok = True
                j2 = 0
                while j2 < k:
                    if chars[start + s + j2] != pattern[j2]:
                        ok = False
                        break
                    j2 += 1
                if ok:
                    found = True
                s += 1
            match[i] = int8(1) if found else int8(0)


def run(offsets, chars, pattern, match, n, k):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](offsets, chars, pattern, match, n, k)
