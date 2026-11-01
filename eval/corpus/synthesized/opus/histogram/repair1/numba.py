"""Task 11, histogram: count the keys by the byte (key >> shift) & 255.

This is the counting pass of one radix sort digit. Each block counts its keys
in a private 256-bin histogram in shared memory, then adds its nonzero bins
to counts in global memory. counts must be zero on entry.

Every loop is a while loop that compares its index with the bound before each
access. A range() loop compiles to a trip-count loop, which does not show that
bound in the code.
"""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK_SIZE: int = 256
BLOCKS_PER_SM: int = 4
BINS: int = 256

TYPES: tuple[types.Type, ...] = (
    types.uint32[:],  # keys, n
    types.int32[:],  # counts, 256
    types.int64,  # shift, 0 to 24
    types.int64,  # n
)


def kernel(
    keys: DeviceNDArray,
    counts: DeviceNDArray,
    shift: int,
    n: int,
) -> None:
    # The ty: ignore comments below: to a host type checker, the Numba CUDA
    # stubs (cuda.threadIdx.x, cuda.blockDim.x, cuda.atomic.add) are Python
    # properties and Stub classes. Numba compiles them to device integers and
    # atomic instructions.
    local = cuda.shared.array(BINS, types.int32)
    tid = cuda.threadIdx.x
    step = cuda.blockDim.x
    b = tid
    while b < BINS:  # ty: ignore[unsupported-operator]
        local[b] = 0
        b += step  # ty: ignore[unsupported-operator]
    cuda.syncthreads()

    # Grid-stride loop over the keys: the grid can be smaller than n.
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        cuda.atomic.add(local, (keys[i] >> shift) & 0xFF, 1)  # ty: ignore[too-many-positional-arguments]
        i += stride
    cuda.syncthreads()

    b = tid
    while b < BINS:  # ty: ignore[unsupported-operator]
        count = local[b]
        if count != 0:
            cuda.atomic.add(counts, b, count)  # ty: ignore[too-many-positional-arguments]
        b += step  # ty: ignore[unsupported-operator]


_kernel = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    counts: DeviceNDArray,
    shift: int,
    n: int,
) -> None:
    """Launch the kernel. A launch needs at least one block, so n == 0 launches nothing.

    The grid is at most BLOCKS_PER_SM blocks per SM, so that the number of
    global atomic adds (256 per block) stays small compared to n.
    """
    if n == 0:
        return
    sm_count = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = min((n + BLOCK_SIZE - 1) // BLOCK_SIZE, sm_count * BLOCKS_PER_SM)
    _kernel[blocks, BLOCK_SIZE](keys, counts, shift, n)
