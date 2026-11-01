from numba import cuda, int32, int64

TYPES = (int64[:], int64[:], int32[:], int64, int64)


def kernel(build, probe, counts, b, p):
    # Lower bound, then upper bound searched from the lower bound.
    stride = cuda.gridDim.x * cuda.blockDim.x
    i = cuda.grid(1)
    while i < p:
        x = probe[i]
        lo = 0
        hi = b
        while lo < hi:
            mid = (lo + hi) >> 1
            if build[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        first = lo
        hi = b
        while lo < hi:
            mid = (lo + hi) >> 1
            if build[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        counts[i] = lo - first
        i += stride


def run(build, probe, counts, b, p):
    blocks = max(1, (p + 255) // 256)
    cuda.jit(kernel)[blocks, 256](build, probe, counts, b, p)
