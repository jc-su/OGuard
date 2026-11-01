from numba import cuda, int32, int64


def kernel(left, right, out, n, r, tolerance):
    # One thread per left row: binary search for the last right row at or
    # before the left value, then apply the tolerance.
    i = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    if i < n:
        x = left[i]
        lo = int64(0)
        hi = r
        while lo < hi:
            mid = (lo + hi) >> 1
            if right[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        res = int32(-1)
        if j >= 0 and x - right[j] <= tolerance:
            res = int32(j)
        out[i] = res


TYPES = (int64[:], int64[:], int32[:], int64, int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(left, right, out, n, r, tolerance):
    blocks = max(1, (n + 255) // 256)
    _jit()[blocks, 256](left, right, out, n, r, tolerance)
