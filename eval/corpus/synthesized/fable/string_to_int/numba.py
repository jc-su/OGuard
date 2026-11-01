from numba import cuda, int32, int64, uint8, uint64


def kernel(offsets, chars, out, valid, n):
    # One thread per byte of the validity bitmap: it parses the eight strings
    # of that byte in turn and writes the byte once, so no atomics on the
    # bitmap are needed. The magnitude accumulates in unsigned arithmetic
    # with the overflow test done before each multiply-add, so that
    # -9223372036854775808 parses and 9223372036854775808 does not.
    b = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    nbytes = (n + 7) // 8
    if b < nbytes:
        bits = int64(0)
        for k in range(8):
            i = b * 8 + k
            if i < n:
                p = int64(offsets[i])
                end = int64(offsets[i + 1])
                ok = False
                value = int64(0)
                if p < end:
                    neg = False
                    c = chars[p]
                    if c == 43 or c == 45:
                        neg = c == 45
                        p += 1
                    if p < end:
                        limit = uint64(0x7FFFFFFFFFFFFFFF)
                        if neg:
                            limit = uint64(0x8000000000000000)
                        mag = uint64(0)
                        ok = True
                        while p < end:
                            c = chars[p]
                            if c < 48 or c > 57:
                                ok = False
                                break
                            d = uint64(c - 48)
                            if mag > (limit - d) // uint64(10):
                                ok = False
                                break
                            mag = mag * uint64(10) + d
                            p += 1
                        if ok:
                            if neg:
                                mag = uint64(0) - mag
                            value = int64(mag)
                out[i] = value
                if ok:
                    bits |= int64(1) << k
        valid[b] = uint8(bits)


TYPES = (int32[:], uint8[:], int64[:], uint8[:], int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(offsets, chars, out, valid, n):
    nbytes = (n + 7) // 8
    blocks = max(1, (nbytes + 255) // 256)
    _jit()[blocks, 256](offsets, chars, out, valid, n)
