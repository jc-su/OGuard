import triton
import triton.language as tl

TYPES = {
    "counts_ptr": "*i32",
    "hist_ptr": "*i32",
    "n_customers": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    counts_ptr,
    hist_ptr,
    n_customers,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs0 = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs0 < n_customers
    cnt = tl.load(counts_ptr + offs0 + 1, mask=mask, other=-1)

    for b in tl.static_range(0, 64):
        hits = mask & (cnt == b)
        addv = tl.sum(tl.where(hits, 1, 0), axis=0).to(tl.int32)
        tl.atomic_add(hist_ptr + b, addv, sem="relaxed")
