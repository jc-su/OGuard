import numpy as np
from numba import cuda, int32, int64, uint8

THREADS = 256
ITEMS = 8
TILE = THREADS * ITEMS


def kernel(values, valid, out, n):
    # One block scans the whole column: the kernel has no scratch memory for a
    # cross-block carry. Thread t owns rows tile + 8 * t .. tile + 8 * t + 7, which
    # are exactly the bits of one validity byte.
    t = cuda.threadIdx.x
    s_has = cuda.shared.array(256, dtype=int32)
    s_val = cuda.shared.array(256, dtype=int64)
    carry = int64(0)
    for tile in range(0, n, 2048):
        base = tile + int64(t) * 8
        bits = 0
        if base < n:
            bits = int32(valid[base >> 3])
        has = int32(0)
        val = int64(0)
        for k in range(8):
            if base + k < n and ((bits >> k) & 1) != 0:
                has = 1
                val = values[base + k]
        # inclusive scan over (has, val); combine(a, b) = b if b.has else a
        s_has[t] = has
        s_val[t] = val
        cuda.syncthreads()
        d = 1
        while d < 256:
            prev_has = int32(0)
            prev_val = int64(0)
            if t >= d:
                prev_has = s_has[t - d]
                prev_val = s_val[t - d]
            cuda.syncthreads()
            if t >= d and s_has[t] == 0:
                s_has[t] = prev_has
                s_val[t] = prev_val
            cuda.syncthreads()
            d *= 2
        state = carry
        if t > 0 and s_has[t - 1] != 0:
            state = s_val[t - 1]
        next_carry = carry
        if s_has[255] != 0:
            next_carry = s_val[255]
        cuda.syncthreads()
        for k in range(8):
            if base + k < n:
                if ((bits >> k) & 1) != 0:
                    state = values[base + k]
                out[base + k] = state
        carry = next_carry


TYPES = (int64[:], uint8[:], int64[:], int64)

_kernel = cuda.jit(kernel)


def run(values, valid, out, n):
    _kernel[1, THREADS](values, valid, out, n)
