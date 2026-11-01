from numba import cuda, int64, uint8


def kernel(source, out, n, offset):
    # One thread per output byte, assembled from the two source bytes that
    # hold its bits.
    b = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    nbytes = (n + 7) // 8
    if b < nbytes:
        src_bytes = (offset + n + 7) // 8
        q = (offset >> 3) + b
        sh = offset & 7
        acc = int64(source[q]) >> sh
        if sh != 0 and q + 1 < src_bytes:
            acc |= int64(source[q + 1]) << (8 - sh)
        rows = n - b * 8
        if rows < 8:
            acc &= (int64(1) << rows) - 1
        out[b] = uint8(acc & 0xFF)


TYPES = (uint8[:], uint8[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(source, out, n, offset):
    nbytes = (n + 7) // 8
    blocks = max(1, (nbytes + 255) // 256)
    _jit()[blocks, 256](source, out, n, offset)
