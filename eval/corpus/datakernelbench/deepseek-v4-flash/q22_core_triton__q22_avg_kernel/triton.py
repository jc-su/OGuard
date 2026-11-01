import triton
import triton.language as tl

TYPES = {
    "acctbal_ptr": "*fp64",
    "in_mask_ptr": "*i1",
    "avg_out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    acctbal_ptr, in_mask_ptr, avg_out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    c_acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    c_in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)

    avg_mask = (c_in_mask == 1) & (c_acctbal > 0.0) & mask
    avg_sum = tl.sum(tl.where(avg_mask, c_acctbal, 0.0))
    avg_count = tl.sum(avg_mask.to(tl.float64))

    tl.atomic_add(avg_out_ptr, avg_sum)
    tl.atomic_add(avg_out_ptr + 1, avg_count)
