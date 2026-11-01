"""Task 12, segmented_sum: sums[j] = sum(values[offsets[j]:offsets[j + 1]]).

The kernel selects one of three mappings from the average segment length
(offsets[g] - offsets[0]) / g. All threads compute the same average, so the
selection is uniform over the grid:

- average >= BLOCK_MODE_MIN_AVERAGE: one block per segment.
- average >= WARP_MODE_MIN_AVERAGE: one warp per segment.
- otherwise: one thread per segment. A segment of WARP_SIZE or more values is
  summed by the whole warp of its thread instead, so that one long segment
  does not stall a warp of short ones.

Each mapping uses a grid-stride loop, so every grid size gives the correct
result; the grid size only changes the speed. An empty segment gives 0.0.
Each segment is summed in a fixed order, so the result is deterministic.

Launch requirement: the block size is a multiple of 32, so that each group of
32 consecutive threads is one hardware warp.
"""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK_SIZE: int = 256
WARP_SIZE: int = 32
FULL_MASK: int = 0xFFFFFFFF
WARP_MODE_MIN_AVERAGE: int = 32
BLOCK_MODE_MIN_AVERAGE: int = 1024

TYPES: tuple[types.Type, ...] = (
    types.int32[:],  # offsets, g + 1
    types.float64[:],  # values, offsets[g]
    types.float64[:],  # sums, g
    types.int64,  # g
)


@cuda.jit(device=True)
def warp_sum(x: float) -> float:
    """Return the sum of x over the warp in lane 0."""
    delta = WARP_SIZE // 2
    while delta > 0:
        x += cuda.shfl_down_sync(FULL_MASK, x, delta)
        delta //= 2
    return x


@cuda.jit(device=True)
def strided_sum(values: DeviceNDArray, i: int, end: int, step: int) -> float:
    """Return the sum of values[i], values[i + step], ... below end.

    Four independent accumulators keep four loads in flight per thread.
    """
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    while i + 3 * step < end:
        a0 += values[i]
        a1 += values[i + step]
        a2 += values[i + 2 * step]
        a3 += values[i + 3 * step]
        i += 4 * step
    while i < end:
        a0 += values[i]
        i += step
    return (a0 + a1) + (a2 + a3)


def kernel(
    offsets: DeviceNDArray,
    values: DeviceNDArray,
    sums: DeviceNDArray,
    g: int,
) -> None:
    # The ty: ignore comments below: to a host type checker, the Numba CUDA
    # stubs (cuda.threadIdx.x, cuda.blockDim.x, cuda.ffs, ...) are Python
    # properties and Stub classes. Numba compiles them to device integers and
    # intrinsics.
    warp_sums = cuda.shared.array(WARP_SIZE, types.float64)
    total = offsets[g] - offsets[0]
    lane = cuda.threadIdx.x % WARP_SIZE  # ty: ignore[unsupported-operator]

    if total >= BLOCK_MODE_MIN_AVERAGE * g:
        # One block per segment: each thread sums a strided part, then the
        # warp sums go through shared memory to warp 0.
        warp_in_block = cuda.threadIdx.x // WARP_SIZE  # ty: ignore[unsupported-operator]
        warps_in_block = cuda.blockDim.x // WARP_SIZE  # ty: ignore[unsupported-operator]
        for j in range(cuda.blockIdx.x, g, cuda.gridDim.x):  # ty: ignore[invalid-argument-type]
            acc = warp_sum(
                strided_sum(values, offsets[j] + cuda.threadIdx.x, offsets[j + 1], cuda.blockDim.x)
            )
            if lane == 0:
                warp_sums[warp_in_block] = acc
            cuda.syncthreads()
            if warp_in_block == 0:
                acc = warp_sum(warp_sums[lane] if lane < warps_in_block else 0.0)
                if lane == 0:
                    sums[j] = acc
            # warp_sums is written again in the next iteration.
            cuda.syncthreads()
    elif total >= WARP_MODE_MIN_AVERAGE * g:
        # One warp per segment. j is the same in all lanes, so all lanes reach warp_sum.
        warp = cuda.grid(1) // WARP_SIZE
        warps = cuda.gridsize(1) // WARP_SIZE
        for j in range(warp, g, warps):
            acc = 0.0
            for i in range(offsets[j] + lane, offsets[j + 1], WARP_SIZE):
                acc += values[i]
            acc = warp_sum(acc)
            if lane == 0:
                sums[j] = acc
    else:
        # One thread per segment. The loop runs over groups of 32 segments,
        # one group per warp, so base is the same in all lanes.
        warp = cuda.grid(1) // WARP_SIZE
        warps = cuda.gridsize(1) // WARP_SIZE
        for base in range(warp * WARP_SIZE, g, warps * WARP_SIZE):
            j = base + lane
            begin = 0
            end = 0
            if j < g:
                begin = offsets[j]
                end = offsets[j + 1]
            is_long = end - begin >= WARP_SIZE
            # The whole warp sums each long segment of the group, one at a time.
            # ballot_sync returns a signed 32-bit value; the mask keeps the 32
            # lane bits and removes the sign extension.
            pending = cuda.ballot_sync(FULL_MASK, is_long) & FULL_MASK
            while pending != 0:
                src = cuda.ffs(pending) - 1  # ty: ignore[unsupported-operator, too-many-positional-arguments]
                pending &= pending - 1
                long_begin = cuda.shfl_sync(FULL_MASK, begin, src)
                long_end = cuda.shfl_sync(FULL_MASK, end, src)
                acc = 0.0
                for i in range(long_begin + lane, long_end, WARP_SIZE):
                    acc += values[i]
                acc = warp_sum(acc)
                if lane == 0:
                    sums[base + src] = acc
            if j < g and not is_long:
                acc = 0.0
                for i in range(begin, end):
                    acc += values[i]
                sums[j] = acc


_kernel = cuda.jit(kernel)


def blocks_for(n: int, g: int) -> int:
    """Return the grid for the mapping that the kernel selects when offsets[0] == 0.

    If offsets[0] > 0, the kernel can select a mapping for shorter segments;
    the grid-stride loops keep that correct.
    """
    warps_per_block = BLOCK_SIZE // WARP_SIZE
    if n >= BLOCK_MODE_MIN_AVERAGE * g:
        return g
    if n >= WARP_MODE_MIN_AVERAGE * g:
        return (g + warps_per_block - 1) // warps_per_block
    return (g + BLOCK_SIZE - 1) // BLOCK_SIZE


def run(
    offsets: DeviceNDArray,
    values: DeviceNDArray,
    sums: DeviceNDArray,
    g: int,
) -> None:
    """Launch the kernel. A launch needs at least one block, so g == 0 launches nothing."""
    if g == 0:
        return
    _kernel[blocks_for(values.shape[0], g), BLOCK_SIZE](offsets, values, sums, g)
