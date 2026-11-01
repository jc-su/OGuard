import triton
import triton.language as tl

TYPES = {
    "c_custkey_ptr": "*i64",
    "counts_ptr": "*i32",
    "hist_ptr": "*i64",
}

CONSTANTS = {"n_customers": 1500000, "counts_len": 1500001, "hist_len": 15000001, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    c_custkey_ptr,
    counts_ptr,
    hist_ptr,
    n_customers: tl.constexpr,
    counts_len: tl.constexpr,
    hist_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers

    cust = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    valid_cust = mask & (cust >= 0) & (cust < counts_len)

    cnt = tl.load(counts_ptr + cust, mask=valid_cust, other=0)
    valid_cnt = valid_cust & (cnt >= 0) & (cnt < hist_len)

    tl.atomic_add(hist_ptr + cnt, 1, sem="relaxed", mask=valid_cnt)
