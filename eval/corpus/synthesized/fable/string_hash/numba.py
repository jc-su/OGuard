from numba import cuda, int32, int64, uint8, uint32


def kernel(offsets, chars, out, n):
    # One thread per string: FNV-1a over its bytes.
    i = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    if i < n:
        start = int64(offsets[i])
        end = int64(offsets[i + 1])
        h = uint32(2166136261)
        for p in range(start, end):
            h = (h ^ uint32(chars[p])) * uint32(16777619)
        out[i] = h


TYPES = (int32[:], uint8[:], uint32[:], int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(offsets, chars, out, n):
    blocks = max(1, (n + 255) // 256)
    _jit()[blocks, 256](offsets, chars, out, n)
