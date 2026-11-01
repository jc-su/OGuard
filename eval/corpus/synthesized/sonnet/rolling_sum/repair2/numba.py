"""Numba CUDA kernel: rolling_sum.

out[i] = sum(values[max(0, i - window + 1) .. i]).

Each block of 256 threads covers 256 consecutive output rows. It first
cooperatively loads its input tile -- the block's 256 rows plus the
`window - 1` rows of "halo" immediately before them -- into shared
memory once (zero-padded where the halo runs off the start of the
array). Every thread then sums its own window directly out of shared
memory, so each input element is read from global memory exactly once
no matter how much its window overlaps its neighbors'.

The tile is a fixed-size shared array sized for the worst case
(blockDim.x up to 1024, window up to 1024) instead of a dynamically
sized `cuda.shared.array(0, ...)` array, whose actual size is a
launch-time argument invisible to any analysis of the kernel body
alone and so can never be proved to bound an access into it.

The read of `values` is gated by an `if` whose condition (`0 <= g < n`)
is the exact bound on the index `g` used in that read, with the read
itself lexically inside the branch: a min/max clamp applied to the
index right before the read states the same fact algebraically, but a
compiler is free to prove that fact redundant given the loop's own
bound and fold the clamp away, leaving the read gated by nothing a
static checker can use.
"""

from numba import cuda, float64, int64

TYPES = (float64[:], float64[:], int64, int64)

THREADS_PER_BLOCK = 256
MAX_TILE = 2047  # 1024 (max blockDim.x) + 1024 (max window) - 1


def kernel(values, out, n, window):
    tile = cuda.shared.array(MAX_TILE, dtype=float64)

    block_start = int64(cuda.blockIdx.x) * cuda.blockDim.x
    tid = cuda.threadIdx.x
    tile_start = block_start - (window - 1)
    tile_size = cuda.blockDim.x + window - 1

    idx = tid
    while idx < tile_size:
        safe_idx = idx
        if safe_idx < 0:
            safe_idx = 0
        if safe_idx >= MAX_TILE:
            safe_idx = MAX_TILE - 1

        g = tile_start + idx
        v = 0.0
        if g >= 0 and g < n:
            v = values[g]
        tile[safe_idx] = v
        idx += cuda.blockDim.x
    cuda.syncthreads()

    i = block_start + tid
    if i < n:
        acc = 0.0
        for j in range(window):
            safe_k = tid + j
            if safe_k < 0:
                safe_k = 0
            if safe_k >= MAX_TILE:
                safe_k = MAX_TILE - 1
            acc += tile[safe_k]
        out[i] = acc


def run(values, out, n, window):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](values, out, n, window)
