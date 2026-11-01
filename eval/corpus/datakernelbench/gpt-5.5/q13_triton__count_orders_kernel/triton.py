import triton
import triton.language as tl

TYPES = {
    "o_custkey_ptr": "*i64",
    "excluded_ptr": "*i1",
    "counts_ptr": "*i32",
}

CONSTANTS = {"n_orders": 15000000, "counts_len": 1500001, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_custkey_ptr,
    excluded_ptr,
    counts_ptr,
    n_orders: tl.constexpr,
    counts_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders

    cust = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    excluded = tl.load(excluded_ptr + offs, mask=mask, other=1)

    keep = mask & (excluded == 0) & (cust >= 0) & (cust < counts_len)
    tl.atomic_add(counts_ptr + cust, 1, sem="relaxed", mask=keep)
