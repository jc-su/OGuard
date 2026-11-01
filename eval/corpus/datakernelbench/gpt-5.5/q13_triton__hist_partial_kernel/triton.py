import triton
import triton.language as tl

TYPES = {
    "c_custkey_ptr": "*i64",
    "counts_ptr": "*i32",
    "partial_hist_ptr": "*i64",
    "overflow_ptr": "*i64",
}

CONSTANTS = {"n_customers": 1500000, "counts_len": 1500001, "SMALL_BINS": 256, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    c_custkey_ptr,
    counts_ptr,
    partial_hist_ptr,
    overflow_ptr,
    n_customers: tl.constexpr,
    counts_len: tl.constexpr,
    SMALL_BINS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers

    cust = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    valid_cust = mask & (cust >= 0) & (cust < counts_len)

    cnt = tl.load(counts_ptr + cust, mask=valid_cust, other=SMALL_BINS)
    in_small = valid_cust & (cnt >= 0) & (cnt < SMALL_BINS)

    hist_idx = pid * SMALL_BINS + cnt
    tl.atomic_add(partial_hist_ptr + hist_idx, 1, sem="relaxed", mask=in_small)

    overflow_lanes = valid_cust & ((cnt < 0) | (cnt >= SMALL_BINS))
    overflow_sum = tl.sum(tl.where(overflow_lanes, 1, 0))
    tl.atomic_add(overflow_ptr, overflow_sum, sem="relaxed")
