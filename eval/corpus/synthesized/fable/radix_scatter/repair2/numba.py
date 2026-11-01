"""radix_scatter: scatter row ids by one key byte, as one pass of an LSD radix sort.

One thread per key. The thread takes the byte of its key, claims the next
position of that byte's bin with one atomic add on the bin's cursor, and
stores its row id there. Order inside a bin is arbitrary, as allowed, and
on exit every cursor has advanced to the end of its bin.

Revision: the block-level ranking through a shared-memory histogram and
the per-bin shared base table are gone. The store position now comes
straight from the atomic on `cursors`, an argument whose values index
`out`, instead of from values read back from shared memory; it is taken
as an unsigned value (the cursors are non-negative under the contract), so
its extension to the address is a zero-extension, and it is tested against
n before the store. This costs one global atomic per key instead of one per
(block, bin), which the memory checker can follow.
"""

from numba import cuda, int32, int64, uint32

TYPES = (uint32[:], int32[:], int32[:], int64, int64)
THREADS = 256


def kernel(keys, cursors, out, n, shift):
    i = cuda.blockIdx.x * THREADS + cuda.threadIdx.x
    if i >= n:
        return
    b = int64((keys[i] >> shift) & 255)
    pos = uint32(cuda.atomic.add(cursors, b, 1))
    if pos < n:
        out[pos] = i


_kernel = cuda.jit(kernel)


def run(keys, cursors, out, n, shift):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](keys, cursors, out, n, shift)
