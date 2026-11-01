import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


def kernel(keys, out, n):
    s = cuda.shared.array(1024, dtype=int32)
    tid = cuda.threadIdx.x
    tile = cuda.blockIdx.x * 1024
    for c in range(4):
        r = tid + c * 256
        if tile + r < n:
            s[r] = keys[tile + r]
        else:
            s[r] = 2147483647
    cuda.syncthreads()
    k = 2
    while k <= 1024:
        j = k >> 1
        while j > 0:
            for c in range(2):
                t = tid + c * 256
                i = ((t & ~(j - 1)) << 1) | (t & (j - 1))
                l = i | j
                x = s[i]
                y = s[l]
                if (i & k) == 0:
                    if x > y:
                        s[i] = y
                        s[l] = x
                else:
                    if x < y:
                        s[i] = y
                        s[l] = x
            cuda.syncthreads()
            j >>= 1
        k <<= 1
    for c in range(4):
        r = tid + c * 256
        if tile + r < n:
            out[tile + r] = s[r]


TYPES = (int32[:], int32[:], int64)
_jit = cuda.jit(kernel)


def run(keys, out, n):
    if n > 0:
        _jit[(n + 1023) // 1024, 256](keys, out, n)
