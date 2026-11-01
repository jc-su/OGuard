"""string_equal: match[i] = 1 when string i of column a equals string i of column b."""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256


def kernel(
    a_offsets: DeviceNDArray,
    a_chars: DeviceNDArray,
    b_offsets: DeviceNDArray,
    b_chars: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
) -> None:
    i = cuda.grid(1)
    if i < n:
        a0 = a_offsets[i]
        b0 = b_offsets[i]
        length = a_offsets[i + 1] - a0
        equal = length == b_offsets[i + 1] - b0
        j = 0
        while equal and j < length:
            equal = a_chars[a0 + j] == b_chars[b0 + j]
            j += 1
        if equal:
            match[i] = 1
        else:
            match[i] = 0


TYPES = (
    types.int32[:],
    types.uint8[:],
    types.int32[:],
    types.uint8[:],
    types.int8[:],
    types.int64,
)

_jitted = cuda.jit(kernel)


def run(
    a_offsets: DeviceNDArray,
    a_chars: DeviceNDArray,
    b_offsets: DeviceNDArray,
    b_chars: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
) -> None:
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _jitted[blocks, THREADS](a_offsets, a_chars, b_offsets, b_chars, match, n)
