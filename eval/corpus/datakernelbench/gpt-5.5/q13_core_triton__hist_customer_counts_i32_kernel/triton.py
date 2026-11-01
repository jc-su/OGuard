import triton
import triton.language as tl

TYPES = {
    "c_custkey_ptr": "*i64",
    "counts_ptr": "*i32",
    "hist_ptr": "*i32",
    "n_customers": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    c_custkey_ptr,
    counts_ptr,
    hist_ptr,
    n_customers,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers
    keys = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    cnt = tl.load(counts_ptr + keys, mask=mask, other=-1)

    for b in tl.static_range(0, 64):
        hits = mask & (cnt == b)
        addv = tl.sum(tl.where(hits, 1, 0), axis=0).to(tl.int32)
        tl.atomic_add(hist_ptr + b, addv, sem="relaxed")
