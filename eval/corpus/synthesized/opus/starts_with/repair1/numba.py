"""Task 10, starts_with: match[i] = 1 when string i starts with the byte pattern prefix.

The string column is in the Arrow layout: string i is the byte range
chars[offsets[i]:offsets[i + 1]]. The comparison is bytewise, so for a valid
UTF-8 pattern it is also a code point prefix test. An empty pattern (k == 0)
matches every string.

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
    types.uint8[:],  # prefix, k
    types.int8[:],  # match, n
    types.int64,  # n
    types.int64,  # k
)


def kernel(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    prefix: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
    k: int,
) -> None:
    # One thread per string, grid-stride loop over the strings.
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        begin = offsets[i]
        result = 0
        if offsets[i + 1] - begin >= k:
            result = 1
            j = 0
            while j < k:
                if chars[begin + j] != prefix[j]:
                    result = 0
                    break
                j += 1
        match[i] = result
        i += stride


_kernel = cuda.jit(kernel)


def run(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    prefix: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
    k: int,
) -> None:
    """Launch the kernel. A launch needs at least one block, so n == 0 launches nothing."""
    if n == 0:
        return
    blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    _kernel[blocks, BLOCK_SIZE](offsets, chars, prefix, match, n, k)
