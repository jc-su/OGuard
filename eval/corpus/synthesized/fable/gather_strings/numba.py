from numba import cuda, int32, int64, uint8


def kernel(offsets, chars, idx, out_offsets, out_chars, n, m):
    # One thread per output byte. The row that owns the byte is found by a
    # binary search over out_offsets, so the work is balanced for any string
    # lengths and the writes are coalesced.
    p = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    total = int64(out_offsets[m])
    if p < total:
        lo = int64(0)
        hi = m
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if int64(out_offsets[mid]) <= p:
                lo = mid
            else:
                hi = mid
        row = int64(idx[lo])
        src = int64(offsets[row]) + (p - int64(out_offsets[lo]))
        out_chars[p] = chars[src]


TYPES = (int32[:], uint8[:], int32[:], int32[:], uint8[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(offsets, chars, idx, out_offsets, out_chars, n, m):
    total = out_chars.shape[0]
    blocks = max(1, (total + 255) // 256)
    _jit()[blocks, 256](offsets, chars, idx, out_offsets, out_chars, n, m)
