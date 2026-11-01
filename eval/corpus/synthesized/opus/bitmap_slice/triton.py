"""bitmap_slice: out bit i = source bit offset + i, for i < n; bits n and above of the
last output byte are 0.

Output byte k takes bits [offset + 8k, offset + 8k + 8): the high part of source
byte offset // 8 + k and, when offset % 8 != 0, the low part of the next byte. The
next byte is read only when it exists: source has (offset + n + 7) // 8 bytes.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {
    "source": "*u8",
    "out": "*u8",
    "n": "i64",
    "offset": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(source, out, n, offset, BLOCK: tl.constexpr):
    n = tl.cast(n, tl.int64)
    offset = tl.cast(offset, tl.int64)
    out_bytes = (n + 7) >> 3
    source_bytes = (offset + n + 7) >> 3
    s = (offset & 7).to(tl.int32)
    tail = (n & 7).to(tl.int32)
    k = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = k < out_bytes
    b = (offset >> 3) + k
    lo = tl.load(source + b, mask=in_range, other=0).to(tl.int32)
    hi = tl.load(source + b + 1, mask=in_range & (s != 0) & (b + 1 < source_bytes), other=0).to(tl.int32)
    v = (lo >> s) | (hi << (8 - s))
    v = tl.where((k == out_bytes - 1) & (tail != 0), v & ((1 << tail) - 1), v)
    tl.store(out + k, (v & 0xFF).to(tl.uint8), mask=in_range)


def run(source, out, n, offset):
    grid = (max(1, triton.cdiv((n + 7) // 8, BLOCK)),)
    kernel[grid](source, out, n, offset, BLOCK=BLOCK, num_warps=NUM_WARPS)
