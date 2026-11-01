from numba import cuda, int32, int64, uint8

TYPES = (int32[:], uint8[:], uint8[:], int64, int64)


def kernel(offsets, chars, out, n, k):
    # One thread per output byte: neighbouring threads copy neighbouring bytes.
    total = n * k
    stride = cuda.gridDim.x * cuda.blockDim.x
    t = cuda.grid(1)
    while t < total:
        i = t // k
        j = t - i * k
        start = offsets[i]
        if j < offsets[i + 1] - start:
            out[t] = chars[start + j]
        else:
            out[t] = 0
        t += stride


def run(offsets, chars, out, n, k):
    blocks = max(1, (n * k + 255) // 256)
    cuda.jit(kernel)[blocks, 256](offsets, chars, out, n, k)
