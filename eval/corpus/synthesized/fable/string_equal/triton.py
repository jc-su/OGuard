import triton
import triton.language as tl


@triton.jit
def kernel(a_offsets, a_chars, b_offsets, b_chars, match, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a_start = tl.load(a_offsets + offs, mask=mask, other=0)
    a_end = tl.load(a_offsets + offs + 1, mask=mask, other=0)
    b_start = tl.load(b_offsets + offs, mask=mask, other=0)
    b_end = tl.load(b_offsets + offs + 1, mask=mask, other=0)
    length = a_end - a_start
    eq = mask & (length == (b_end - b_start))
    longest = tl.max(tl.where(eq, length, 0), axis=0)
    for j in range(0, longest):
        live = eq & (j < length)
        ca = tl.load(a_chars + (a_start + j), mask=live, other=0)
        cb = tl.load(b_chars + (b_start + j), mask=live, other=0)
        eq = eq & (ca == cb)
    tl.store(match + offs, eq.to(tl.int8), mask=mask)


TYPES = {
    "a_offsets": "*i32",
    "a_chars": "*u8",
    "b_offsets": "*i32",
    "b_chars": "*u8",
    "match": "*i8",
    "n": "i64",
}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(a_offsets, a_chars, b_offsets, b_chars, match, n):
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](
        a_offsets,
        a_chars,
        b_offsets,
        b_chars,
        match,
        n,
        BLOCK=CONSTANTS["BLOCK"],
        num_warps=NUM_WARPS,
    )
