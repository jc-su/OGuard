from numba import cuda, int32, int64

BLOCK = 256
TILE = 1024


def kernel(values, out, n):
    # Row i of tile t takes the value at the mirrored position of the same tile. Consecutive threads
    # read consecutive addresses in reverse order, which coalesces as well as a forward copy.
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    for i in range(start, n, stride):
        base = i - i % TILE
        length = min(TILE, n - base)
        out[i] = values[base + length - 1 - (i - base)]


TYPES = (int32[:], int32[:], int64)


def run(values, out, n):
    blocks = max(1, -(-n // BLOCK))
    cuda.jit(kernel)[blocks, BLOCK](values, out, n)
