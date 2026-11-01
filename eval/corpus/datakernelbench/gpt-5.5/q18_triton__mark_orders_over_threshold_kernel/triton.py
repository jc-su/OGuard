import triton
import triton.language as tl

TYPES = {
    "orderkey_ptr": "*i64",
    "sum_by_key_ptr": "*fp64",
    "threshold_ptr": "*fp64",
    "out_sum_ptr": "*fp64",
    "out_keep_ptr": "*i1",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    orderkey_ptr,
    sum_by_key_ptr,
    threshold_ptr,
    out_sum_ptr,
    out_keep_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    threshold = tl.load(threshold_ptr)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keys = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    sums = tl.load(sum_by_key_ptr + keys, mask=mask, other=0.0)
    keep = sums > threshold

    tl.store(out_sum_ptr + offs, sums, mask=mask)
    tl.store(out_keep_ptr + offs, keep, mask=mask)
