import numpy as np
from numba import cuda
from numba.types import int32, int64


def kernel(values, row, n):
    sv = cuda.shared.array(256, int64)
    si = cuda.shared.array(256, int32)
    t = cuda.threadIdx.x

    bv = np.int64(0)
    bi = np.int32(-1)
    stride = np.int64(cuda.gridDim.x) * 256
    i = np.int64(cuda.blockIdx.x) * 256 + t
    while i < n:
        v = values[i]
        if bi < 0 or v < bv:
            bv = v
            bi = np.int32(i)
        i += stride
    sv[t] = bv
    si[t] = bi
    cuda.syncthreads()

    s = 128
    while s > 0:
        if t < s:
            ov = sv[t + s]
            oi = si[t + s]
            # The other candidate wins with a smaller value, or an equal value at a smaller row.
            if oi >= 0 and (si[t] < 0 or ov < sv[t] or (ov == sv[t] and oi < si[t])):
                sv[t] = ov
                si[t] = oi
        cuda.syncthreads()
        s >>= 1

    # row[0] holds the row of the current minimum, so values supplies its value.
    if t == 0 and si[0] >= 0:
        cv = sv[0]
        ci = si[0]
        cur = row[0]
        while True:
            if cur >= 0:
                v = values[cur]
                if v < cv or (v == cv and cur < ci):
                    break
            old = cuda.atomic.cas(row, 0, cur, ci)
            if old == cur:
                break
            cur = old


TYPES = (int64[:], int32[:], int64)

_kernel = cuda.jit(kernel)


def run(values, row, n):
    blocks = cuda.get_current_device().MULTIPROCESSOR_COUNT * 8
    grid = max(1, min((n + 255) // 256, blocks))
    _kernel[grid, 256](values, row, n)
