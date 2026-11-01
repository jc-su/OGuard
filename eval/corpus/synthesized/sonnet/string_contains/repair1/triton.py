import triton
import triton.language as tl

# One CTA handles one string. `k` (pattern length) is fixed for the whole
# launch, so the inner chunk loop over the pattern unrolls at compile time
# for short patterns and otherwise becomes a small number of vectorized
# chunk comparisons; each of the BLOCK_K lanes checks one pattern byte in
# parallel, so only the (per-row, data-dependent) shift count is a serial
# GPU loop.
BLOCK_K = 128


@triton.jit
def kernel(offsets, chars, pattern, match, n, k, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    # The launch uses grid=(n,), so `row` is always < n for the intended
    # launch, but the kernel itself must not depend on that: for an
    # adversarial (oversized) grid, `row` can reach or exceed `n`, and
    # every access below is indexed by `row`. Guard explicitly instead of
    # relying on the caller's grid choice.
    if row >= n:
        return

    if k == 0:
        # The empty pattern occurs in every string.
        tl.store(match + row, 1)
        return

    start = tl.load(offsets + row)
    end = tl.load(offsets + row + 1)
    str_len = end - start
    max_start = str_len - k + 1

    found = tl.zeros((), dtype=tl.int32)
    for s in range(0, max_start):
        mismatch = tl.zeros((), dtype=tl.int32)
        for c0 in range(0, k, BLOCK_K):
            offs_k = c0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < k
            pat = tl.load(pattern + offs_k, mask=kmask, other=0)
            strv = tl.load(chars + start + s + offs_k, mask=kmask, other=0)
            neq = tl.where(kmask, (pat != strv).to(tl.int32), 0)
            mismatch += tl.sum(neq, axis=0)
        this_match = (mismatch == 0).to(tl.int32)
        found = tl.maximum(found, this_match)

    tl.store(match + row, found.to(tl.int8))


TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "pattern": "*u8",
    "match": "*i8",
    "n": "i64",
    "k": "i64",
}
CONSTANTS = {"BLOCK_K": BLOCK_K}
NUM_WARPS = 4


def run(offsets, chars, pattern, match, n, k):
    grid = (n,)
    kernel[grid](offsets, chars, pattern, match, n, k, BLOCK_K=BLOCK_K, num_warps=NUM_WARPS)
