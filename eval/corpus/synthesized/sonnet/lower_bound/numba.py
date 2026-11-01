from numba import cuda, int32, int64

BLOCK = 256


def kernel(keys, sorted, out, n, s):
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    for i in range(start, n, stride):
        key = keys[i]
        lo = int64(0)
        hi = s
        while lo < hi:
            mid = (lo + hi) >> 1
            if sorted[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo


TYPES = (int64[:], int64[:], int32[:], int64, int64)


def run(keys, sorted, out, n, s):
    blocks = max(1, -(-n // BLOCK))
    cuda.jit(kernel)[blocks, BLOCK](keys, sorted, out, n, s)
