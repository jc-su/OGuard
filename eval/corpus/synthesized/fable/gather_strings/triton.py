import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, idx, out_offsets, out_chars, n, m, BLOCK: tl.constexpr):
    # Each program writes BLOCK output bytes. The row that owns a byte is found
    # by a binary search over out_offsets, so the work is balanced for any
    # string lengths and the writes are coalesced.
    pid = tl.program_id(0).to(tl.int64)
    p = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    total = tl.load(out_offsets + m).to(tl.int64)
    pmask = p < total
    lo = tl.zeros([BLOCK], tl.int64)
    hi = tl.where(pmask, m, 1) + tl.zeros([BLOCK], tl.int64)
    while tl.max(hi - lo, axis=0) > 1:
        mid = (lo + hi) >> 1
        v = tl.load(out_offsets + mid, mask=pmask, other=0).to(tl.int64)
        take = v <= p
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid)
    row = tl.load(idx + lo, mask=pmask, other=0).to(tl.int64)
    start = tl.load(out_offsets + lo, mask=pmask, other=0).to(tl.int64)
    src = tl.load(offsets + row, mask=pmask, other=0).to(tl.int64) + (p - start)
    c = tl.load(chars + src, mask=pmask, other=0)
    tl.store(out_chars + p, c, mask=pmask)


TYPES = {"offsets": "*i32", "chars": "*u8", "idx": "*i32", "out_offsets": "*i32",
         "out_chars": "*u8", "n": "i64", "m": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(offsets, chars, idx, out_offsets, out_chars, n, m):
    total = out_chars.shape[0]
    grid = (max(1, triton.cdiv(total, CONSTANTS["BLOCK"])),)
    kernel[grid](offsets, chars, idx, out_offsets, out_chars, n, m,
                 BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
