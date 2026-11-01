"""ascii_upper: copy a string column's bytes to out with a-z replaced by A-Z.

The column's bytes are chars[offsets[0]:offsets[n]]. The threads convert one
byte each per step of a grid-stride loop over that range. run sizes the grid
for about 16 bytes per thread, since one byte per thread launches so many
blocks that block scheduling limits the throughput.
"""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
BYTES_PER_THREAD = 16


def kernel(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    end = offsets[n]
    stride = cuda.gridsize(1)
    i = offsets[0] + cuda.grid(1)
    while i < end:
        c = chars[i]
        if c >= 97 and c <= 122:
            out[i] = c - 32
        else:
            out[i] = c
        i += stride


TYPES = (
    types.int32[:],
    types.uint8[:],
    types.uint8[:],
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    # out holds offsets[n] bytes, so its length sizes the grid without
    # reading offsets back to the host.
    per_block = THREADS * BYTES_PER_THREAD
    blocks = (out.shape[0] + per_block - 1) // per_block
    if blocks > 0:
        _kernel[blocks, THREADS](offsets, chars, out, n)
