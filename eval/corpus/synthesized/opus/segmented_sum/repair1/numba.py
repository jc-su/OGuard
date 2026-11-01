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

Every access to values uses an index i with offsets[j] <= i < offsets[j + 1]
for a segment j < g, read from offsets in the same thread. The warp that sums
a long segment of the one-thread mapping reads offsets[j] and offsets[j + 1]
of that segment again, instead of receiving them from the lane of the segment
with a shuffle. Every loop is a while loop that compares its index with the
bound before each access; a range() loop compiles to a trip-count loop, which
does not show that bound in the code. The warp indices use shifts and masks
of nonnegative thread indices, not the floor division and remainder of
signed integers. The device functions have forceinline=True, so the kernel
has no calls.

Launch requirement: the block size is a multiple of 32, so that each group of
32 consecutive threads is one hardware warp.
"""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK_SIZE: int = 256
WARP_SIZE: int = 32
LOG_WARP_SIZE: int = 5
MAX_WARPS: int = 32  # Warps in a block of 1024 threads, the largest block.
FULL_MASK: int = 0xFFFFFFFF
WARP_MODE_MIN_AVERAGE: int = 32
BLOCK_MODE_MIN_AVERAGE: int = 1024

TYPES: tuple[types.Type, ...] = (
    types.int32[:],  # offsets, g + 1
    types.float64[:],  # values, offsets[g]
    types.float64[:],  # sums, g
    types.int64,  # g
)


# The signatures of cuda.jit that ty sees do not name forceinline, which
# numba-cuda accepts through its keyword arguments, so both device functions
# carry a ty: ignore comment.
@cuda.jit(device=True, forceinline=True)  # ty: ignore[unknown-argument]
def warp_sum(x: float) -> float:
    """Return the sum of x over the warp in lane 0."""
    delta = WARP_SIZE // 2
    while delta > 0:
        x += cuda.shfl_down_sync(FULL_MASK, x, delta)
        delta //= 2
    return x


@cuda.jit(device=True, forceinline=True)  # ty: ignore[unknown-argument]
def strided_sum(values: DeviceNDArray, i: int, end: int, step: int) -> float:
    """Return the sum of values[i], values[i + step], ... below end.

    Four independent accumulators keep four loads in flight per thread. step
    is positive, so i + step, i + 2 * step, and i + 3 * step are at most
    i + 3 * step, which the loop compares with end.
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
    warp_sums = cuda.shared.array(MAX_WARPS, types.float64)
    total = offsets[g] - offsets[0]
    tid = cuda.threadIdx.x
    lane = tid & (WARP_SIZE - 1)  # ty: ignore[unsupported-operator]

    if total >= BLOCK_MODE_MIN_AVERAGE * g:
        # One block per segment: each thread sums a strided part, then the
        # warp sums go through shared memory to warp 0. A block has at most
        # 1024 threads, so tid >> LOG_WARP_SIZE is below MAX_WARPS; the mask
        # does not change it and keeps the index inside warp_sums.
        warp_in_block = (tid >> LOG_WARP_SIZE) & (MAX_WARPS - 1)  # ty: ignore[unsupported-operator]
        warps_in_block = cuda.blockDim.x >> LOG_WARP_SIZE  # ty: ignore[unsupported-operator]
        j = cuda.blockIdx.x
        while j < g:  # ty: ignore[unsupported-operator]
            acc = warp_sum(
                strided_sum(values, offsets[j] + tid, offsets[j + 1], cuda.blockDim.x)  # ty: ignore[unsupported-operator]
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
            j += cuda.gridDim.x  # ty: ignore[unsupported-operator]
    elif total >= WARP_MODE_MIN_AVERAGE * g:
        # One warp per segment. j is the same in all lanes, so all lanes reach warp_sum.
        warps = cuda.gridsize(1) >> LOG_WARP_SIZE
        j = cuda.grid(1) >> LOG_WARP_SIZE
        while j < g:
            end = offsets[j + 1]
            i = offsets[j] + lane
            acc = 0.0
            while i < end:
                acc += values[i]
                i += WARP_SIZE
            acc = warp_sum(acc)
            if lane == 0:
                sums[j] = acc
            j += warps
    else:
        # One thread per segment. The loop runs over groups of 32 segments,
        # one group per warp, so base is the same in all lanes.
        warps = cuda.gridsize(1) >> LOG_WARP_SIZE
        base = (cuda.grid(1) >> LOG_WARP_SIZE) << LOG_WARP_SIZE
        while base < g:
            j = base + lane
            begin = 0
            end = 0
            if j < g:
                begin = offsets[j]
                end = offsets[j + 1]
            is_long = end - begin >= WARP_SIZE
            # The whole warp sums each long segment of the group, one at a
            # time. pending and long_j are the same in all lanes, so all lanes
            # reach warp_sum. ballot_sync returns a signed 32-bit value; the
            # mask keeps the 32 lane bits and removes the sign extension. The
            # ballot sets only bits of lanes with j < g, so the check
            # long_j < g is always true; it keeps the reads of offsets inside
            # offsets.
            pending = cuda.ballot_sync(FULL_MASK, is_long) & FULL_MASK
            while pending != 0:
                src = types.uint32(cuda.ffs(pending) - 1)  # ty: ignore[unsupported-operator, too-many-positional-arguments]
                pending &= pending - 1
                long_j = base + src
                if long_j < g:
                    long_end = offsets[long_j + 1]
                    i = offsets[long_j] + lane
                    acc = 0.0
                    while i < long_end:
                        acc += values[i]
                        i += WARP_SIZE
                    acc = warp_sum(acc)
                    if lane == 0:
                        sums[long_j] = acc
            if j < g and not is_long:
                i = begin
                acc = 0.0
                while i < end:
                    acc += values[i]
                    i += 1
                sums[j] = acc
            base += warps << LOG_WARP_SIZE


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
