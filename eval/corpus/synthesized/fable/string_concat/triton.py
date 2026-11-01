import triton
import triton.language as tl


@triton.jit
def kernel(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n, BLOCK: tl.constexpr):
    # Each program writes BLOCK output bytes. The row that owns a byte is found
    # by a binary search over out_offsets; the byte comes from the first column
    # when its position in the row is below that row's first length, and from
    # the second column otherwise.
    pid = tl.program_id(0).to(tl.int64)
    p = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    total = tl.load(out_offsets + n).to(tl.int64)
    pmask = p < total
    lo = tl.zeros([BLOCK], tl.int64)
    hi = tl.where(pmask, n, 1) + tl.zeros([BLOCK], tl.int64)
    while tl.max(hi - lo, axis=0) > 1:
        mid = (lo + hi) >> 1
        v = tl.load(out_offsets + mid, mask=pmask, other=0).to(tl.int64)
        take = v <= p
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid)
    local = p - tl.load(out_offsets + lo, mask=pmask, other=0).to(tl.int64)
    a0 = tl.load(a_offsets + lo, mask=pmask, other=0).to(tl.int64)
    la = tl.load(a_offsets + lo + 1, mask=pmask, other=0).to(tl.int64) - a0
    b0 = tl.load(b_offsets + lo, mask=pmask, other=0).to(tl.int64)
    from_a = pmask & (local < la)
    ca = tl.load(a_chars + a0 + local, mask=from_a, other=0)
    cb = tl.load(b_chars + b0 + (local - la), mask=pmask & (local >= la), other=0)
    tl.store(out_chars + p, tl.where(from_a, ca, cb), mask=pmask)


TYPES = {"a_offsets": "*i32", "a_chars": "*u8", "b_offsets": "*i32", "b_chars": "*u8",
         "out_offsets": "*i32", "out_chars": "*u8", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    total = out_chars.shape[0]
    grid = (max(1, triton.cdiv(total, CONSTANTS["BLOCK"])),)
    kernel[grid](a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n,
                 BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
