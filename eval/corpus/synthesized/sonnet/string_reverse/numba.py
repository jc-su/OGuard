from numba import cuda, int32, int64, uint8

BLOCK = 256


def kernel(offsets, chars, out, n):
    # One thread per output byte. The string that owns byte p is the last s with offsets[s] <= p,
    # so bytes are spread evenly over threads however the string lengths are distributed.
    total = int64(offsets[n])
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    for p in range(start, total, stride):
        lo = int64(0)
        hi = n + 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if offsets[mid] <= p:
                lo = mid + 1
            else:
                hi = mid
        s = lo - 1
        out[p] = chars[int64(offsets[s]) + int64(offsets[s + 1]) - 1 - p]


TYPES = (int32[:], uint8[:], uint8[:], int64)


def run(offsets, chars, out, n):
    total = int(offsets[n])
    blocks = max(1, -(-total // BLOCK))
    cuda.jit(kernel)[blocks, BLOCK](offsets, chars, out, n)
