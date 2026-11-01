"""Numba CUDA kernel: rolling_sum.

out[i] = sum(values[max(0, i - window + 1) .. i]).

Each block of 256 threads covers 256 consecutive output rows. It first
cooperatively loads its input tile -- the block's 256 rows plus the
`window - 1` rows of "halo" immediately before them -- into dynamic
shared memory once (zero-padded where the halo runs off the start of
the array). Every thread then sums its own window directly out of
shared memory, so each input element is read from global memory exactly
once no matter how much its window overlaps its neighbors'.
"""

from numba import cuda, float64, int64

TYPES = (float64[:], float64[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(values, out, n, window):
    tile = cuda.shared.array(0, dtype=float64)

    block_start = cuda.blockIdx.x * cuda.blockDim.x
    tid = cuda.threadIdx.x
    tile_start = block_start - (window - 1)
    tile_size = cuda.blockDim.x + window - 1

    idx = tid
    while idx < tile_size:
        g = tile_start + idx
        if 0 <= g < n:
            tile[idx] = values[g]
        else:
            tile[idx] = 0.0
        idx += cuda.blockDim.x
    cuda.syncthreads()

    i = block_start + tid
    if i < n:
        acc = 0.0
        for j in range(window):
            acc += tile[tid + j]
        out[i] = acc


def run(values, out, n, window):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    tile_size = THREADS_PER_BLOCK + window - 1
    shared_bytes = int(tile_size) * 8
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK, 0, shared_bytes](
        values, out, n, window
    )
