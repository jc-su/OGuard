"""bitmap_slice: out bit i = source bit offset + i, for i < n; bits n and above of the
last output byte are 0.

Output byte k takes bits [offset + 8k, offset + 8k + 8): the high part of source
byte offset // 8 + k and, when offset % 8 != 0, the low part of the next byte. The
next byte is read only when it exists: source has (offset + n + 7) // 8 bytes.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32

TYPES = (
    types.uint8[:],  # source
    types.uint8[:],  # out
    types.int64,  # n
    types.int64,  # offset
)


def kernel(source, out, n, offset):
    out_bytes = (n + 7) >> 3
    source_bytes = (offset + n + 7) >> 3
    first = offset >> 3
    s = offset & 7
    tail = n & 7
    k = cuda.grid(1)
    stride = cuda.gridsize(1)
    while k < out_bytes:
        b = first + k
        v = types.int64(source[b]) >> s
        if s != 0 and b + 1 < source_bytes:
            v |= types.int64(source[b + 1]) << (8 - s)
        if k == out_bytes - 1 and tail != 0:
            v &= (1 << tail) - 1
        out[k] = v & 0xFF
        k += stride


_kernel = cuda.jit(kernel)


def run(source, out, n, offset):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    out_bytes = (n + 7) // 8
    blocks = max(1, min((out_bytes + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](source, out, n, offset)
