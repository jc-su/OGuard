"""histogram: counts[(key >> shift) & 255] += 1, one radix-sort digit pass.

Each block accumulates a private 256-bin histogram in shared memory (fast
shared-memory atomics), then flushes its non-zero bins to the global counts
with one atomicAdd per bin per block.

counts must be zero on entry.
"""

from numba import cuda, int32, int64, uint32

THREADS = 256
# A small grid keeps the per-block flush (256 global atomics per block) cheap
# relative to the shared-memory work; 256 blocks still fill every SM.
MAX_BLOCKS = 256
BINS = 256


def kernel(keys, counts, shift, n):
    local = cuda.shared.array(BINS, int32)
    tid = cuda.threadIdx.x
    j = tid
    while j < BINS:
        local[j] = 0
        j += cuda.blockDim.x
    cuda.syncthreads()

    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        cuda.atomic.add(local, (keys[i] >> shift) & 255, 1)
        i += stride
    cuda.syncthreads()

    j = tid
    while j < BINS:
        c = local[j]
        if c != 0:
            cuda.atomic.add(counts, j, c)
        j += cuda.blockDim.x


TYPES = (uint32[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(keys, counts, shift, n):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](keys, counts, shift, n)
