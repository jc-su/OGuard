from numba import cuda, int8, int64

BLOCK = 256


def kernel(values, flags, n):
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    for i in range(start, n, stride):
        if i == 0:
            flags[i] = 1
        elif values[i] != values[i - 1]:
            flags[i] = 1
        else:
            flags[i] = 0


TYPES = (int64[:], int8[:], int64)


def run(values, flags, n):
    blocks = max(1, -(-n // BLOCK))
    cuda.jit(kernel)[blocks, BLOCK](values, flags, n)
