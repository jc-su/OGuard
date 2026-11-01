import triton
import triton.language as tl

TYPES = {
    "acctbal_ptr": "*fp64",
    "cntrycode_ptr": "*i32",
    "in_mask_ptr": "*i32",
    "avg_out_ptr": "*fp64",
    "count_out_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    acctbal_ptr, cntrycode_ptr, in_mask_ptr,
    avg_out_ptr, count_out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    
    keep = mask & (in_mask > 0) & (acctbal > 0.0)
    count = tl.sum(tl.where(keep, 1, 0))
    sum_bal = tl.sum(tl.where(keep, acctbal, 0.0))
    
    if count > 0:
        tl.atomic_add(avg_out_ptr, sum_bal)
        tl.atomic_add(count_out_ptr, count)
