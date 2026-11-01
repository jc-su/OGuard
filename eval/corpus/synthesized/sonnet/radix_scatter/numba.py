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
    i = cuda.grid(1)
    if i < n:
        byte = (keys[i] >> shift) & 255
        pos = cuda.atomic.add(cursors, byte, 1)
        out[pos] = int32(i)


def run(keys, cursors, out, n, shift):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](keys, cursors, out, n, shift)
