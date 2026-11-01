"""Numba CUDA kernel: radix_scatter.

The scatter pass of an LSD radix sort pass: `cursors[b]` already holds
the exclusive prefix sum of the counts of each byte value `b` (from a
prior histogram pass). One thread per row extracts its key's byte at
`shift`, claims the next free slot for that byte with `atomicAdd`
(advancing the cursor), and writes its row id there. Contention is
spread across 256 independent counters, so this is far cheaper than a
single global compaction counter; a random uint32 key distributes its
rows roughly evenly over the 256 buckets.
"""

from numba import cuda, int32, int64, uint32

TYPES = (uint32[:], int32[:], int32[:], int64, int64)

THREADS_PER_BLOCK = 256


def kernel(keys, cursors, out, n, shift):
    # Compute the row index in explicit 64-bit arithmetic: cuda.grid(1)
    # combines blockIdx/blockDim/threadIdx in 32-bit registers, and for an
    # adversarial (oversized) grid that product can overflow 32 bits.
    i = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i >= 0 and i < n:
        byte = (keys[i] >> shift) & 255
        pos = cuda.atomic.add(cursors, byte, 1)
        # `pos` is the pre-increment value of a bucket cursor, not a
        # direct function of the thread index; it is bounded by `n` only
        # through the global invariant that `cursors` starts as a correct
        # histogram prefix sum over every row (part of this task's
        # contract), which is not something a per-thread bound check can
        # see. Guard the write against `out`'s declared length directly
        # so the store's address is provably in range for any input.
        if pos >= 0 and pos < n:
            out[pos] = int32(i)


def run(keys, cursors, out, n, shift):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](keys, cursors, out, n, shift)
