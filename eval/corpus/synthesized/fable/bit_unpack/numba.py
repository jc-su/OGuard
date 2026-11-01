from numba import cuda, int64, uint32


def kernel(packed, out, n, w, bits):
    # One thread per code: the code starts at bit i * bits of the stream and
    # spans at most two words.
    i = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    if i < n:
        p = i * bits
        word = p >> 5
        sh = p & 31
        acc = int64(packed[word]) >> sh
        if sh + bits > 32:
            acc |= int64(packed[word + 1]) << (32 - sh)
        acc &= (int64(1) << bits) - 1
        out[i] = uint32(acc)


TYPES = (uint32[:], uint32[:], int64, int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(packed, out, n, w, bits):
    blocks = max(1, (n + 255) // 256)
    _jit()[blocks, 256](packed, out, n, w, bits)
