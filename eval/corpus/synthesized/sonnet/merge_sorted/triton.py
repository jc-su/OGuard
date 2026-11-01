import triton
import triton.language as tl

BLOCK_SIZE = 256
ITERS = 32


@triton.jit
def kernel(a, b, out, m, p, BLOCK_SIZE: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = m + p
    mask = idx < total
    is_a = idx < m

    i = idx
    j = idx - m

    a_val = tl.load(a + i, mask=mask & is_a, other=0)
    b_val = tl.load(b + j, mask=mask & (~is_a), other=0)
    val = tl.where(is_a, a_val, b_val)

    # Merge path: an a-element's output slot is its own rank in `a` plus
    # the count of b-elements strictly less than it (so ties in b land
    # after it); a b-element's slot is its own rank in `b` plus the count
    # of a-elements less than or equal to it (so ties in a land before
    # it). Every element computes its final position independently via
    # binary search, so the whole merge is embarrassingly parallel.
    lo_b = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    hi_b = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + p
    for _ in range(ITERS):
        active = mask & (lo_b < hi_b)
        mid = (lo_b + hi_b) // 2
        bv = tl.load(b + mid, mask=active, other=0)
        cond = active & (bv < val)
        lo_b = tl.where(cond, mid + 1, lo_b)
        hi_b = tl.where(active & (~cond), mid, hi_b)
    count_b = lo_b

    lo_a = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    hi_a = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + m
    for _ in range(ITERS):
        active = mask & (lo_a < hi_a)
        mid = (lo_a + hi_a) // 2
        av = tl.load(a + mid, mask=active, other=0)
        cond = active & (av <= val)
        lo_a = tl.where(cond, mid + 1, lo_a)
        hi_a = tl.where(active & (~cond), mid, hi_a)
    count_a = lo_a

    pos = tl.where(is_a, i + count_b, j + count_a)
    tl.store(out + pos, val, mask=mask)


TYPES = {
    "a": "*i64",
    "b": "*i64",
    "out": "*i64",
    "m": "i64",
    "p": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE, "ITERS": ITERS}
NUM_WARPS = 8


def run(a, b, out, m, p):
    total = m + p
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    kernel[grid](a, b, out, m, p, BLOCK_SIZE=BLOCK_SIZE, ITERS=ITERS, num_warps=NUM_WARPS)
