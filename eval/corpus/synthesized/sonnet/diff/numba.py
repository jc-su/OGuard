from numba import cuda, int64

BLOCK = 256


def kernel(values, out, n):
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    for i in range(start, n, stride):
        if i == 0:
            out[i] = values[i]
        else:
            out[i] = values[i] - values[i - 1]


TYPES = (int64[:], int64[:], int64)


def run(values, out, n):
    blocks = max(1, -(-n // BLOCK))
    cuda.jit(kernel)[blocks, BLOCK](values, out, n)
