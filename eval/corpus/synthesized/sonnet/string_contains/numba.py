"""Numba CUDA kernel: string_contains.

match[i] = 1 if `pattern` occurs as a byte substring of string i, else 0.
The empty pattern matches every string.

One thread per row does a naive (Sunday/brute-force) substring scan of
its own string. `pattern` is small and read by every thread in the
block, so it is staged into dynamic shared memory once per block (a
cooperative strided copy) instead of every thread re-reading it from
global memory for every row and every candidate start offset.
"""

from numba import cuda, int8, int32, int64, uint8

TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(offsets, chars, pattern, match, n, k):
    shared_pattern = cuda.shared.array(0, dtype=uint8)
    tid = cuda.threadIdx.x
    j = tid
    while j < k:
        shared_pattern[j] = pattern[j]
        j += cuda.blockDim.x
    cuda.syncthreads()

    i = cuda.grid(1)
    if i < n:
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
                    if chars[start + s + j2] != shared_pattern[j2]:
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
    shared_bytes = int(k)
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK, 0, shared_bytes](
        offsets, chars, pattern, match, n, k
    )
