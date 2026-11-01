"""Numba CUDA kernel for histogram.

Counts keys by the byte (key >> shift) & 255, one pass of a radix
sort. Each block keeps a private histogram in shared memory (indexed
1:1 by thread id, since BLOCK == 256 bins) and flushes it to the
global counts with one atomic add per bin per block, instead of one
atomic add per key. The grid is capped at MAX_BLOCKS and each thread
strides over multiple keys, so the flush cost does not grow with n.

Precondition: the caller zeroes counts before each launch.
"""

from numba import cuda, int32, int64, uint32

TYPES = (uint32[:], int32[:], int64, int64)

BLOCK = 256
NUM_BINS = 256
MAX_BLOCKS = 1024


def kernel(keys, counts, shift, n):
    tid = cuda.threadIdx.x

    local = cuda.shared.array(NUM_BINS, dtype=int32)
    local[tid] = 0
    cuda.syncthreads()

    i = cuda.grid(1)
    stride = cuda.gridDim.x * cuda.blockDim.x
    while i < n:
        b = int32((keys[i] >> shift) & 255)
        cuda.atomic.add(local, b, 1)
        i += stride
    cuda.syncthreads()

    cuda.atomic.add(counts, tid, local[tid])


def run(keys, counts, shift, n):
    """Launch kernel on device arrays. counts must already be zeroed."""
    if n == 0:
        return
    blocks = min((n + BLOCK - 1) // BLOCK, MAX_BLOCKS)
    cuda.jit(kernel)[blocks, BLOCK](keys, counts, shift, n)
