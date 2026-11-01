from numba import cuda, float64, int32, int64

BLOCK = 256


def kernel(idx, values, out, m, n):
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    for i in range(start, m, stride):
        cuda.atomic.add(out, idx[i], values[i])


TYPES = (int32[:], float64[:], float64[:], int64, int64)


def run(idx, values, out, m, n):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min(-(-m // BLOCK), 8 * sms))
    cuda.jit(kernel)[blocks, BLOCK](idx, values, out, m, n)
