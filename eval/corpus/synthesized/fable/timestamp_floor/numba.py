from numba import cuda, int64


def kernel(ts, out, n, unit):
    # One thread per row. The remainder is taken on a nonnegative magnitude,
    # so a timestamp before 1970 rounds toward negative infinity.
    i = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    if i < n:
        t = ts[i]
        if t >= 0:
            out[i] = t - t % unit
        else:
            r = (-t) % unit
            if r == 0:
                out[i] = t
            else:
                out[i] = t - (unit - r)


TYPES = (int64[:], int64[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(ts, out, n, unit):
    blocks = max(1, (n + 255) // 256)
    _jit()[blocks, 256](ts, out, n, unit)
