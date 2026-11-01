import triton
import triton.language as tl

BLOCK_SIZE = 256
# 2**ITERS comfortably exceeds any run count r we will ever binary-search
# over (r <= n and n fits an int32), so a fixed iteration count keeps the
# loop trip count a compile-time constant without ever truncating a search.
ITERS = 32


@triton.jit
def kernel(run_values, run_ends, out, r, n, BLOCK_SIZE: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row < n

    # Binary search each row for the first run whose run_ends[j] > row,
    # i.e. an upper_bound over run_ends. Search state (lo, hi) freezes once
    # it converges (the predicate is false at the fixed point), so running
    # extra iterations beyond what's needed is harmless.
    lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    hi = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + r

    for _ in range(ITERS):
        active = mask & (lo < hi)
        mid = (lo + hi) // 2
        end_val = tl.load(run_ends + mid, mask=active, other=0)
        cond = active & (end_val <= row)
        lo = tl.where(cond, mid + 1, lo)
        hi = tl.where(active & (~cond), mid, hi)

    j = lo
    val = tl.load(run_values + j, mask=mask, other=0.0)
    tl.store(out + row, val, mask=mask)


TYPES = {
    "run_values": "*fp64",
    "run_ends": "*i32",
    "out": "*fp64",
    "r": "i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE, "ITERS": ITERS}
NUM_WARPS = 8


def run(run_values, run_ends, out, r, n):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](run_values, run_ends, out, r, n, BLOCK_SIZE=BLOCK_SIZE, ITERS=ITERS, num_warps=NUM_WARPS)
