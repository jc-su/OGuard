from numba import cuda, int8, int64, uint64


def kernel(values, out, overflow, n, shift):
    # One thread per row. The arithmetic runs on the unsigned magnitude so
    # that -2^63 and the overflow test need no special cases.
    i = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    if i < n:
        v = values[i]
        neg = v < 0
        mag = uint64(v)
        if neg:
            mag = uint64(0) - mag
        if shift >= 0:
            p = uint64(1)
            for _ in range(shift):
                p = p * uint64(10)
            limit = uint64(0x7FFFFFFFFFFFFFFF)
            if neg:
                limit = uint64(0x8000000000000000)
            if mag > limit // p:
                out[i] = 0
                overflow[i] = int8(1)
            else:
                r = mag * p
                if neg:
                    r = uint64(0) - r
                out[i] = int64(r)
                overflow[i] = int8(0)
        else:
            q = uint64(1)
            for _ in range(-shift):
                q = q * uint64(10)
            qt = mag // q
            rem = mag - qt * q
            if rem * uint64(2) >= q:
                qt = qt + uint64(1)
            if neg:
                qt = uint64(0) - qt
            out[i] = int64(qt)
            overflow[i] = int8(0)


TYPES = (int64[:], int64[:], int8[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(values, out, overflow, n, shift):
    blocks = max(1, (n + 255) // 256)
    _jit()[blocks, 256](values, out, overflow, n, shift)
