from numba import cuda, int8, int64


def kernel(values, out, overflow, n, shift):
    i = int64(cuda.blockIdx.x) * 256 + int64(cuda.threadIdx.x)
    if i >= n:
        return
    k = shift if shift >= 0 else -shift
    pw = int64(1)
    for _ in range(k):
        pw *= 10
    v = values[i]
    if shift >= 0:
        lim = int64(9223372036854775807) // pw
        if pw != 1 and (v > lim or v < -lim):
            out[i] = 0
            overflow[i] = 1
        else:
            out[i] = v * pw
            overflow[i] = 0
    else:
        # floor division: Numba's // and % already floor for a positive divisor
        q = v // pw
        r = v - q * pw
        twice = 2 * r
        if v >= 0:
            up = twice >= pw
        else:
            up = twice > pw
        out[i] = q + 1 if up else q
        overflow[i] = 0


TYPES = (int64[:], int64[:], int8[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(values, out, overflow, n, shift):
    grid = max(1, (n + 255) // 256)
    _kernel[grid, 256](values, out, overflow, n, shift)
