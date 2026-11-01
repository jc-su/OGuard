import numpy as np
from numba import cuda, int32, int64

TYPES = (int64[:], int32[:], int32[:], int64, int64)


def kernel(keys, part, counts, n, p):
    # Per-block histogram in shared memory, merged into the global counts once
    # per block.
    hist = cuda.shared.array(1024, int32)
    tid = cuda.threadIdx.x
    for k in range(tid, p, 256):
        hist[k] = 0
    cuda.syncthreads()
    mult = np.uint64(0x9E3779B97F4A7C15)
    shift = np.uint64(32)
    modulus = np.uint64(p)
    stride = cuda.gridDim.x * cuda.blockDim.x
    i = cuda.grid(1)
    while i < n:
        h = (np.uint64(keys[i]) * mult) >> shift
        k = int32(h % modulus)
        part[i] = k
        cuda.atomic.add(hist, k, 1)
        i += stride
    cuda.syncthreads()
    for k in range(tid, p, 256):
        c = hist[k]
        if c > 0:
            cuda.atomic.add(counts, k, c)


def run(keys, part, counts, n, p):
    blocks = min(1024, max(1, (n + 255) // 256))
    cuda.jit(kernel)[blocks, 256](keys, part, counts, n, p)
