"""ascii_upper: copy the column's bytes with ASCII a-z replaced by A-Z."""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256


def kernel(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    total = offsets[n]
    stride = cuda.gridsize(1)
    j = cuda.grid(1)
    while j < total:
        c = chars[j]
        if c >= 97 and c <= 122:
            c = c - 32
        out[j] = c
        j += stride


TYPES = (
    types.int32[:],
    types.uint8[:],
    types.uint8[:],
    types.int64,
)

_jitted = cuda.jit(kernel)


def run(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    # chars holds offsets[n] bytes, so its size is the number of work items.
    blocks = max(1, (chars.size + THREADS - 1) // THREADS)
    _jitted[blocks, THREADS](offsets, chars, out, n)
