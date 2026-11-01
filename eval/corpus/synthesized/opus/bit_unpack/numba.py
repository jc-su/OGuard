"""bit_unpack: out[i] = the bits-wide code at bit position i * bits of the LSB-first
stream in packed (Parquet bit-packed dictionary codes).

Code i starts at bit p = i * bits (up to 2^35, so 64-bit), in word p // 32 at shift
p % 32. The two words p // 32 and p // 32 + 1 form a 64-bit window that holds the
whole code; the second word is read only when the code reaches into it, which also
keeps the read inside the w words. The window arithmetic is unsigned 64-bit.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32

TYPES = (
    types.uint32[:],  # packed
    types.uint32[:],  # out
    types.int64,  # n
    types.int64,  # w
    types.int64,  # bits
)


def kernel(packed, out, n, w, bits):
    mask = types.uint64(0xFFFFFFFF)
    if bits < 32:
        mask = (types.uint64(1) << types.uint64(bits)) - types.uint64(1)
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        p = i * bits
        word = p >> 5
        shift = p & 31
        window = types.uint64(packed[word])
        if shift + bits > 32:
            window |= types.uint64(packed[word + 1]) << types.uint64(32)
        out[i] = types.uint32((window >> types.uint64(shift)) & mask)
        i += stride


_kernel = cuda.jit(kernel)


def run(packed, out, n, w, bits):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](packed, out, n, w, bits)
