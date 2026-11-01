import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(values, out, n, window, BLOCK_SIZE: tl.constexpr):
    # Compute the row index in 64-bit arithmetic: program_id and arange
    # are 32-bit by default, and for an adversarial (oversized) grid
    # `pid * BLOCK_SIZE` can overflow a 32-bit accumulator, defeating the
    # `row < n` mask.
    pid = tl.program_id(0).to(tl.int64)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    row_mask = row < n

    # window is uniform across the whole launch, so every program executes
    # the same (dynamic-length) trip count; each iteration issues one
    # coalesced, shifted vector load, giving good memory reuse across
    # iterations without needing a scan/halo scheme for a runtime window.
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    for offset in range(0, window):
        idx = row - offset
        in_range = row_mask & (idx >= 0) & (idx < n)
        # Address `values` with `idx` itself -- the exact value whose
        # bounds `in_range` checks -- rather than a separately clamped
        # substitute produced by `tl.where`. A checker proving this load
        # is inside `values` needs the predicate guarding it to mention,
        # directly, the value added to the pointer; it cannot follow a
        # clamp performed on a different value that never reaches the
        # load.
        v = tl.load(values + idx, mask=in_range, other=0.0)
        acc += v

    tl.store(out + row, acc, mask=row_mask)


TYPES = {
    "values": "*fp64",
    "out": "*fp64",
    "n": "i64",
    "window": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE}
NUM_WARPS = 8


def run(values, out, n, window):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](values, out, n, window, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
