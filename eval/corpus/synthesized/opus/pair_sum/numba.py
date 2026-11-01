"""Pair sum: out[i] = left[left_idx[i]] + right[right_idx[i]]."""

from numba import cuda, float64, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int32[:], int32[:], float64[:], float64[:], float64[:], int64, int64, int64)

THREADS = 256


def kernel(left_idx, right_idx, left, right, out, m, l, r):  # noqa: E741 (the task names this argument l)
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    while i < m:
        out[i] = left[left_idx[i]] + right[right_idx[i]]
        i += stride


_KERNEL = cuda.jit(kernel)


def run(
    left_idx: DeviceNDArray,
    right_idx: DeviceNDArray,
    left: DeviceNDArray,
    right: DeviceNDArray,
    out: DeviceNDArray,
    m: int,
    l: int,  # noqa: E741 (the task names this argument l)
    r: int,
) -> None:
    if m == 0:
        return
    blocks = (m + THREADS - 1) // THREADS
    _KERNEL[blocks, THREADS](left_idx, right_idx, left, right, out, m, l, r)
