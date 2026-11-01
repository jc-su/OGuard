"""between_bitmap: out bit i = valid(i) and lo <= values[i] <= hi, as an Arrow bitmap.

Each warp evaluates 32 consecutive rows, assembles their bits with a ballot, and
lanes 0..3 store the four output bytes. A warp's rows start at a multiple of 32,
so each output byte has one writer and the bits at positions n and above are 0.
"""

from numba import cuda, types

THREADS = 256
MAX_BLOCKS_PER_SM = 32

TYPES = (
    types.int64[:],  # values
    types.uint8[:],  # valid
    types.uint8[:],  # out
    types.int64,  # n
    types.int64,  # lo
    types.int64,  # hi
)


def kernel(values, valid, out, n, lo, hi):
    lane = cuda.threadIdx.x & 31
    warp = cuda.grid(1) >> 5
    num_warps = cuda.gridsize(1) >> 5
    out_bytes = (n + 7) >> 3
    base = warp * 32
    while base < n:
        i = base + lane
        bit = False
        if i < n:
            v = values[i]
            bit = ((valid[i >> 3] >> (i & 7)) & 1) != 0 and lo <= v and v <= hi
        word = cuda.ballot_sync(0xFFFFFFFF, bit)
        byte = (base >> 3) + lane
        if lane < 4 and byte < out_bytes:
            out[byte] = (word >> (8 * lane)) & 0xFF
        base += num_warps * 32


_kernel = cuda.jit(kernel)


def run(values, valid, out, n, lo, hi):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * MAX_BLOCKS_PER_SM))
    _kernel[blocks, THREADS](values, valid, out, n, lo, hi)
