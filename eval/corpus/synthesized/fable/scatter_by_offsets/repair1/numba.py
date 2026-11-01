"""scatter_by_offsets: out[offsets[i]] = i for every i with flags[i] != 0.

offsets is the exclusive prefix sum of flags, so the selected rows land in
increasing row order. Grid-stride loop, one row per iteration.

The position is read as an unsigned value and compared against total before
the store, so every store stays inside out (total slots) for any offsets.
Under the contract the comparison always holds for a flagged row, because
its exclusive prefix sum is below the sum of all flags.
"""

from numba import cuda, int8, int32, int64, uint32

THREADS = 256
MAX_BLOCKS = 2048


def kernel(flags, offsets, out, n, total):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        if flags[i] != 0:
            pos = uint32(offsets[i])
            if pos < total:
                out[pos] = i
        i += stride


TYPES = (int8[:], int32[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(flags, offsets, out, n, total):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](flags, offsets, out, n, total)
