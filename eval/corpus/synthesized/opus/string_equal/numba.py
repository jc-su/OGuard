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
        a_begin = a_offsets[i]
        b_begin = b_offsets[i]
        length = a_offsets[i + 1] - a_begin
        equal = length == b_offsets[i + 1] - b_begin
        k = 0
        while equal and k < length:
            equal = a_chars[a_begin + k] == b_chars[b_begin + k]
            k += 1
        match[i] = 1 if equal else 0


TYPES = (
    types.int32[:],
    types.uint8[:],
    types.int32[:],
    types.uint8[:],
    types.int8[:],
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    a_offsets: DeviceNDArray,
    a_chars: DeviceNDArray,
    b_offsets: DeviceNDArray,
    b_chars: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
) -> None:
    blocks = (n + THREADS - 1) // THREADS
    if blocks > 0:
        _kernel[blocks, THREADS](a_offsets, a_chars, b_offsets, b_chars, match, n)
