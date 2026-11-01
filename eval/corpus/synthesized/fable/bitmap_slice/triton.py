import triton
import triton.language as tl


@triton.jit
def kernel(source, out, n, offset, BLOCK: tl.constexpr):
    # Each program writes BLOCK output bytes, each assembled from the two
    # source bytes that hold its bits.
    pid = tl.program_id(0).to(tl.int64)
    b = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    nbytes = (n + 7) // 8
    bmask = b < nbytes
    src_bytes = (b * 0 + offset + n + 7) // 8
    q = b + (offset >> 3)
    sh = (b * 0 + offset) & 7
    first = tl.load(source + q, mask=bmask, other=0).to(tl.int32)
    second = tl.load(source + q + 1, mask=bmask & (sh != 0) & (q + 1 < src_bytes), other=0).to(tl.int32)
    acc = (first >> sh.to(tl.int32)) | (second << (8 - sh).to(tl.int32))
    rows = n - b * 8
    keep = tl.where(rows < 8, (1 << tl.minimum(tl.maximum(rows, 0), 8).to(tl.int32)) - 1, 255)
    tl.store(out + b, (acc & keep).to(tl.uint8), mask=bmask)


TYPES = {"source": "*u8", "out": "*u8", "n": "i64", "offset": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(source, out, n, offset):
    nbytes = (n + 7) // 8
    grid = (max(1, triton.cdiv(nbytes, CONSTANTS["BLOCK"])),)
    kernel[grid](source, out, n, offset, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
