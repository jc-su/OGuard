import triton
import triton.language as tl

TYPES = {
    "c_custkey_ptr": "*i64",
    "order_counts_ptr": "*i64",
    "c_count_out_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 1024,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    c_custkey_ptr,
    order_counts_ptr,
    c_count_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    c_keys = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    counts = tl.load(order_counts_ptr + c_keys, mask=mask, other=0)

    tl.store(c_count_out_ptr + offs, counts, mask=mask)
