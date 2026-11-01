"""bit_unpack: out[i] = the bits-wide code at bit position i * bits of the LSB-first
stream in packed (Parquet bit-packed dictionary codes).

Code i starts at bit p = i * bits (up to 2^35, so 64-bit), in word p // 32 at shift
p % 32. The two words p // 32 and p // 32 + 1 form a 64-bit window that holds the
whole code; the second word is loaded only when the code reaches into it, which
also keeps the load inside the w words.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {
    "packed": "*u32",
    "out": "*u32",
    "n": "i64",
    "w": "i64",
    "bits": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(packed, out, n, w, bits, BLOCK: tl.constexpr):
    bits = tl.cast(bits, tl.int64)
    mask = tl.where(bits >= 32, 0xFFFFFFFF, (tl.full((), 1, tl.int64) << bits) - 1).to(tl.uint64)
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    p = rows * bits
    word = p >> 5
    shift = p & 31
    low = tl.load(packed + word, mask=in_range, other=0).to(tl.uint64)
    high = tl.load(packed + word + 1, mask=in_range & (shift + bits > 32), other=0).to(tl.uint64)
    window = low | (high << 32)
    code = (window >> shift.to(tl.uint64)) & mask
    tl.store(out + rows, code.to(tl.uint32), mask=in_range)


def run(packed, out, n, w, bits):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](packed, out, n, w, bits, BLOCK=BLOCK, num_warps=NUM_WARPS)
