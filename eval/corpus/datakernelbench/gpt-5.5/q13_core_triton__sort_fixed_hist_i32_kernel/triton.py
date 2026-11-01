import triton
import triton.language as tl

TYPES = {
    "hist_ptr": "*i32",
    "out_counts_ptr": "*i64",
    "out_dists_ptr": "*i64",
    "n_unique_ptr": "*i64",
}

CONSTANTS = {"HIST_BINS": 64}

NUM_WARPS = 2


@triton.jit
def kernel(
    hist_ptr,
    out_counts_ptr,
    out_dists_ptr,
    n_unique_ptr,
    HIST_BINS: tl.constexpr,
):
    offs = tl.arange(0, HIST_BINS)
    vals = tl.load(hist_ptr + offs)
    present = vals > 0
    n_unique = tl.sum(tl.where(present, 1, 0), axis=0).to(tl.int64)
    tl.store(n_unique_ptr, n_unique)

    scores = vals * HIST_BINS + offs
    scores = tl.where(present, scores, -1)
    sorted_scores = tl.sort(scores, descending=True)

    dists = sorted_scores // HIST_BINS
    counts = sorted_scores - dists * HIST_BINS
    tl.store(out_counts_ptr + offs, counts.to(tl.int64), mask=offs < n_unique)
    tl.store(out_dists_ptr + offs, dists.to(tl.int64), mask=offs < n_unique)
