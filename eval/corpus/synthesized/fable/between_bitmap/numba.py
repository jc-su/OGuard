from numba import cuda, int64, uint8


def kernel(values, valid, out, n, lo, hi):
    # One thread per output byte: eight consecutive rows.
    b = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    nbytes = (n + 7) // 8
    if b < nbytes:
        base = b * 8
        rows = n - base
        if rows > 8:
            rows = 8
        vbits = int64(valid[b])
        acc = int64(0)
        for k in range(rows):
            if (vbits >> k) & 1:
                x = values[base + k]
                if x >= lo and x <= hi:
                    acc |= int64(1) << k
        out[b] = uint8(acc)


TYPES = (int64[:], uint8[:], uint8[:], int64, int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(values, valid, out, n, lo, hi):
    nbytes = (n + 7) // 8
    blocks = max(1, (nbytes + 255) // 256)
    _jit()[blocks, 256](values, valid, out, n, lo, hi)
