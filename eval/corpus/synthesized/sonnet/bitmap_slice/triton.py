import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n", "offset"])
def kernel(source, out, n, offset, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    j = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    nbytes = (n + 7) // 8
    m = j < nbytes
    s = offset >> 3
    b = (offset & 7).to(tl.int32)
    source_bytes = (offset + n + 7) // 8
    low = tl.load(source + s + j, mask=m, other=0).to(tl.int32)
    has_next = m & (b != 0) & (s + j + 1 < source_bytes)
    high = tl.load(source + s + j + 1, mask=has_next, other=0).to(tl.int32)
    value = ((low >> b) | (high << (8 - b))) & 0xFF
    remaining = tl.minimum(n - 8 * j, 8).to(tl.int32)
    keep = (1 << remaining) - 1
    tl.store(out + j, (value & keep).to(tl.uint8), mask=m)


TYPES = {
    "source": "*u8",
    "out": "*u8",
    "n": "i64",
    "offset": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(source, out, n, offset):
    nbytes = (n + 7) // 8
    grid = (max(1, triton.cdiv(nbytes, CONSTANTS["BLOCK"])),)
    kernel[grid](source, out, n, offset, **CONSTANTS, num_warps=NUM_WARPS)
