from numba import cuda, float64, int32, int64

TYPES = (int32[:], int32[:], float64[:], float64[:], float64[:], int64, int64, int64)


def kernel(left_idx, right_idx, left, right, out, m, l, r):
    stride = cuda.gridDim.x * cuda.blockDim.x
    i = cuda.grid(1)
    while i < m:
        out[i] = left[left_idx[i]] + right[right_idx[i]]
        i += stride


def run(left_idx, right_idx, left, right, out, m, l, r):
    cuda.jit(kernel)[max(1, (m + 255) // 256), 256](left_idx, right_idx, left, right, out, m, l, r)
