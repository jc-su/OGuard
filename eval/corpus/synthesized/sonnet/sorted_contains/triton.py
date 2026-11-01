import triton
import triton.language as tl

BLOCK_SIZE = 256
ITERS = 32


@triton.jit
def kernel(keys, members, match, n, s, BLOCK_SIZE: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    key = tl.load(keys + offs, mask=mask, other=0)

    # Binary search for the insertion point of `key` in `members`
    # (lower_bound). (lo, hi) freeze once converged, so extra iterations
    # beyond what's needed are harmless.
    lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    hi = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + s

    for _ in range(ITERS):
        active = mask & (lo < hi)
        mid = (lo + hi) // 2
        mv = tl.load(members + mid, mask=active, other=0)
        cond = active & (mv < key)
        lo = tl.where(cond, mid + 1, lo)
        hi = tl.where(active & (~cond), mid, hi)

    idx = lo
    idx_valid = mask & (idx < s)
    mv_final = tl.load(members + idx, mask=idx_valid, other=0)
    found = idx_valid & (mv_final == key)

    tl.store(match + offs, found.to(tl.int8), mask=mask)


TYPES = {
    "keys": "*i64",
    "members": "*i64",
    "match": "*i8",
    "n": "i64",
    "s": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE, "ITERS": ITERS}
NUM_WARPS = 8


def run(keys, members, match, n, s):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](keys, members, match, n, s, BLOCK_SIZE=BLOCK_SIZE, ITERS=ITERS, num_warps=NUM_WARPS)
