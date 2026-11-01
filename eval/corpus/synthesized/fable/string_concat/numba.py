from numba import cuda, int32, int64, uint8


def kernel(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    # One thread per output byte. The row that owns the byte is found by a
    # binary search over out_offsets; the byte comes from the first column
    # when its position in the row is below that row's first length, and
    # from the second column otherwise.
    p = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    total = int64(out_offsets[n])
    if p < total:
        lo = int64(0)
        hi = n
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if int64(out_offsets[mid]) <= p:
                lo = mid
            else:
                hi = mid
        local = p - int64(out_offsets[lo])
        a0 = int64(a_offsets[lo])
        la = int64(a_offsets[lo + 1]) - a0
        if local < la:
            out_chars[p] = a_chars[a0 + local]
        else:
            out_chars[p] = b_chars[int64(b_offsets[lo]) + (local - la)]


TYPES = (int32[:], uint8[:], int32[:], uint8[:], int32[:], uint8[:], int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    total = out_chars.shape[0]
    blocks = max(1, (total + 255) // 256)
    _jit()[blocks, 256](a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n)
