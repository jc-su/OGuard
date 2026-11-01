"""String prefix: write the first min(k, length) bytes of each string, padded with 0."""

from numba import cuda, int32, int64, uint8
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int32[:], uint8[:], uint8[:], int64, int64)

THREADS = 256
MAX_K = 64


def kernel(offsets, chars, out, n, k):
    # One thread per output byte, so the stores of a warp are contiguous.
    total = n * k
    stride = cuda.gridsize(1)
    j = cuda.grid(1)
    while j < total:
        i = j // k
        c = j - i * k
        start = offsets[i]
        if c < offsets[i + 1] - start:
            out[j] = chars[start + c]
        else:
            out[j] = 0
        j += stride


_KERNEL = cuda.jit(kernel)


def run(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
    k: int,
) -> None:
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be in [1, {MAX_K}], got {k}")
    total = n * k
    if total == 0:
        return
    blocks = (total + THREADS - 1) // THREADS
    _KERNEL[blocks, THREADS](offsets, chars, out, n, k)
