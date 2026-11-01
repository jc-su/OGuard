"""Stream compaction: write the values of the kept rows to the front of out."""

from numba import cuda, int8, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int64[:], int8[:], int64[:], int32[:], int64)

THREADS = 256
FULL_MASK = 0xFFFFFFFF


def kernel(values, keep, out, count, n):
    lane = cuda.laneid
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    # i - lane is the first row of the warp and is equal in all 32 lanes, so all
    # lanes run the same iterations and the full-mask warp intrinsics are valid.
    while i - lane < n:
        flag = i < n and keep[i] != 0
        mask = cuda.ballot_sync(FULL_MASK, flag)
        # One atomic per warp reserves the output slots of all kept lanes.
        base = 0
        if lane == 0 and mask != 0:
            base = cuda.atomic.add(count, 0, cuda.popc(mask))
        base = cuda.shfl_sync(FULL_MASK, base, 0)
        if flag:
            out[base + cuda.popc(mask & cuda.lanemask_lt())] = values[i]
        i += stride


_KERNEL = cuda.jit(kernel)


def run(
    values: DeviceNDArray,
    keep: DeviceNDArray,
    out: DeviceNDArray,
    count: DeviceNDArray,
    n: int,
) -> None:
    if n == 0:
        return
    blocks = (n + THREADS - 1) // THREADS
    _KERNEL[blocks, THREADS](values, keep, out, count, n)
