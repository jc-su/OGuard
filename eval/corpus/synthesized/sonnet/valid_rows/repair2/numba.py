"""Numba CUDA kernel: valid_rows.

Compacts the row ids whose bit is set in an Arrow validity bitmap into
`rows`, and writes how many there are into `count[0]`.

The previous revision assigned one thread per row and derived that
thread's byte index as `byte_idx = i // 8`, guarded by
`byte_idx < valid_len` with `valid_len = (n + 7) // 8` recomputed
locally to match the buffer's declared length. That guard is true for
every `i < n` -- `i < n` implies `i // 8 < (n + 7) // 8` by construction
-- and a compiler is free to use exactly that implication to prove the
inner guard redundant given the outer `i < n` check and fold it away,
which erases the one condition that ties the read of `valid` to its
own buffer's length, leaving nothing for a static checker to use.

This revision instead assigns one thread per BYTE of the bitmap, so the
byte index is the thread's own grid position, not a value derived from
a separately-bounded row index: `byte_idx < valid_len` is the sole
condition under which anything happens in the thread at all, so it is
not implied by, or foldable into, any other condition in the kernel.
Each thread then decodes its own byte's up to 8 rows and reserves its
output slots with one atomicAdd per byte (rather than one per valid
row), still far fewer atomics than the naive one-per-row scheme even
though it forgoes full warp-aggregated ballots.
"""

from numba import cuda, int32, int64, uint8

TYPES = (uint8[:], int32[:], int32[:], int64)

THREADS_PER_BLOCK = 256


def kernel(valid, rows, count, n):
    valid_len = (n + 7) // 8
    # byte_idx is this thread's own grid position: nothing else in the
    # kernel implies `byte_idx < valid_len`, so a compiler has no other
    # fact from which to derive (and then discard) this guard.
    byte_idx = int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if byte_idx >= 0 and byte_idx < valid_len:
        byte = valid[byte_idx]
        base = byte_idx * 8

        local_count = int32(0)
        for bit in range(8):
            row = base + bit
            if row < n and ((byte >> bit) & 1) != 0:
                local_count += 1

        if local_count > 0:
            offset = cuda.atomic.add(count, 0, local_count)
            slot = int32(0)
            for bit in range(8):
                row = base + bit
                if row < n and ((byte >> bit) & 1) != 0:
                    rows[offset + slot] = int32(row)
                    slot += 1


def run(valid, rows, count, n):
    if n <= 0:
        return
    valid_len = (n + 7) // 8
    blocks = (valid_len + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](valid, rows, count, n)
