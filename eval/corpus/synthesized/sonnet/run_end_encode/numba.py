import numpy as np
from numba import cuda
from numba.types import int32, int64


# One block walks the column tile by tile. A thread owns 8 consecutive rows per
# tile. The number of run starts before a row is the position of its run, so a
# scan of the per-thread start counts, with a carry over the tiles, gives every
# run its output position. Work is linear for any run length.
def kernel(values, run_ends, run_values, count, n):
    scan = cuda.shared.array(256, int64)
    t = cuda.threadIdx.x
    carry = np.int64(0)
    for tile in range(0, n, 2048):
        base = tile + t * 8
        c = np.int64(0)
        for k in range(8):
            i = base + k
            if i < n and (i == 0 or values[i] != values[i - 1]):
                c += 1
        scan[t] = c
        cuda.syncthreads()
        d = 1
        while d < 256:
            other = np.int64(0)
            if t >= d:
                other = scan[t - d]
            cuda.syncthreads()
            scan[t] += other
            cuda.syncthreads()
            d *= 2
        pos = carry + scan[t] - c
        tile_total = scan[255]
        # The run that starts at row i has index pos and closes the run pos - 1.
        for k in range(8):
            i = base + k
            if i < n and (i == 0 or values[i] != values[i - 1]):
                run_values[pos] = values[i]
                if pos > 0:
                    run_ends[pos - 1] = i
                pos += 1
        carry += tile_total
        cuda.syncthreads()
    # The last run ends at row n.
    if t == 0 and n > 0:
        run_ends[carry - 1] = n
        count[0] = carry


TYPES = (int64[:], int32[:], int64[:], int32[:], int64)

_kernel = cuda.jit(kernel)


def run(values, run_ends, run_values, count, n):
    _kernel[1, 256](values, run_ends, run_values, count, n)
