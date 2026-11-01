import numpy as np
from numba import cuda, int32, int64, uint8

THREADS = 256


def kernel(offsets, chars, out, valid, n):
    # One thread per string. The validity bits of the 32 rows of a warp are
    # combined with a ballot, and lanes 0 to 3 each write one byte of the bitmap.
    tid = cuda.threadIdx.x
    lane = tid & 31
    i = np.int64(cuda.blockIdx.x) * 256 + tid
    ok = False
    value = np.int64(0)
    if i < n:
        p = np.int64(offsets[i])
        end = np.int64(offsets[i + 1])
        negative = False
        if p < end:
            first = chars[p]
            if first == 43:
                p += 1
            elif first == 45:
                negative = True
                p += 1
        limit = np.uint64(9223372036854775807)
        if negative:
            limit += np.uint64(1)
        acc = np.uint64(0)
        ok = p < end
        while ok and p < end:
            digit = np.int64(chars[p]) - 48
            if digit < 0 or digit > 9:
                ok = False
            else:
                d = np.uint64(digit)
                if acc > (limit - d) // np.uint64(10):
                    ok = False
                else:
                    acc = acc * np.uint64(10) + d
                    p += 1
        if ok:
            value = np.int64(acc)
            if negative:
                value = -value
        out[i] = value
    bits = np.int64(cuda.ballot_sync(0xFFFFFFFF, ok))
    byte = ((np.int64(cuda.blockIdx.x) * 256 + tid - lane) >> 3) + lane
    if lane < 4 and byte < (n + 7) >> 3:
        valid[byte] = (bits >> (8 * lane)) & 255


TYPES = (int32[:], uint8[:], int64[:], uint8[:], int64)

_kernel = cuda.jit(kernel)


def run(offsets, chars, out, valid, n):
    grid = max(1, (n + THREADS - 1) // THREADS)
    _kernel[grid, THREADS](offsets, chars, out, valid, n)
