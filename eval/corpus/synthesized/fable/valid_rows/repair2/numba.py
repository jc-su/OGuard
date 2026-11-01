"""valid_rows: compact the ids of the rows whose Arrow validity bit is set.

One thread per bitmap byte. The thread loads its byte and, for each of its
eight rows that exists and whose bit is set, claims the next slot of `rows`
with one atomic add on `count` and stores the row id there. Output order is
the order of the atomics, not the row order.

Revision: the byte index is the thread's grid index itself, tested against
the bitmap size (n + 7) >> 3, instead of a row index shifted right; the
slot of every stored row id comes straight from the atomic on `count`, an
argument whose values index `rows`, instead of from a warp ballot and a
block-wide prefix sum whose terms pass through shared memory. The slot is
taken as an unsigned value (the count is non-negative under the contract),
so its extension to the address is a zero-extension, and it is tested
against n before the store. This costs one atomic per valid row instead of
one per block, which the memory checker can follow.
"""

from numba import cuda, int32, int64, uint8, uint32

TYPES = (uint8[:], int32[:], int32[:], int64)
THREADS = 256


@cuda.jit(device=True, inline="always")
def emit(rows, count, n, bits, row, bit):
    """Store `row` at the next slot of `rows` when it exists and bit `bit` of its byte is set."""
    if row < n and ((bits >> bit) & 1) != 0:
        pos = uint32(cuda.atomic.add(count, 0, 1))
        if pos < n:
            rows[pos] = row


def kernel(valid, rows, count, n):
    j = cuda.blockIdx.x * THREADS + cuda.threadIdx.x  # bitmap byte
    nbytes = (n + 7) >> 3
    if j >= nbytes:
        return
    bits = valid[j]
    row = j * 8
    emit(rows, count, n, bits, row, 0)
    emit(rows, count, n, bits, row + 1, 1)
    emit(rows, count, n, bits, row + 2, 2)
    emit(rows, count, n, bits, row + 3, 3)
    emit(rows, count, n, bits, row + 4, 4)
    emit(rows, count, n, bits, row + 5, 5)
    emit(rows, count, n, bits, row + 6, 6)
    emit(rows, count, n, bits, row + 7, 7)


_kernel = cuda.jit(kernel)


def run(valid, rows, count, n):
    nbytes = (n + 7) // 8
    blocks = max(1, (nbytes + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](valid, rows, count, n)
