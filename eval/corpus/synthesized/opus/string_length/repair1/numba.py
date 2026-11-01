"""Task 9, string_length: the number of UTF-8 code points of each string.

The string column is in the Arrow layout: string i is the byte range
chars[offsets[i]:offsets[i + 1]]. A code point starts at every byte that is
not a continuation byte (0b10xxxxxx), so the kernel counts those bytes.

Every loop is a while loop that compares its index with the bound before each
access. A range() loop compiles to a trip-count loop, which does not show that
bound in the code.
"""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK_SIZE: int = 256

TYPES: tuple[types.Type, ...] = (
    types.int32[:],  # offsets, n + 1
    types.uint8[:],  # chars, offsets[n]
    types.int32[:],  # lengths, n
    types.int64,  # n
)


def kernel(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    lengths: DeviceNDArray,
    n: int,
) -> None:
    # One thread per string, grid-stride loop over the strings.
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        count = 0
        j = offsets[i]
        end = offsets[i + 1]
        while j < end:
            if (chars[j] & 0xC0) != 0x80:
                count += 1
            j += 1
        lengths[i] = count
        i += stride


_kernel = cuda.jit(kernel)


def run(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    lengths: DeviceNDArray,
    n: int,
) -> None:
    """Launch the kernel. A launch needs at least one block, so n == 0 launches nothing."""
    if n == 0:
        return
    blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    _kernel[blocks, BLOCK_SIZE](offsets, chars, lengths, n)
