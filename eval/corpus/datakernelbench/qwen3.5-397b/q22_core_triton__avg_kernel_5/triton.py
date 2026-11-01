import triton
import triton.language as tl

TYPES = {
    "acctbal_ptr": "*fp64",
    "in_mask_ptr": "*i1",
    "sum_ptr": "*fp64",
    "count_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 4


@triton.jit
def kernel(
    acctbal_ptr, in_mask_ptr,
    sum_ptr, count_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    in_m = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    
    # Condition: in_mask & bal > 0
    cond = in_m & (bal > 0.0)
    
    # Sum and Count
    local_sum = tl.sum(tl.where(cond, bal, 0.0))
    local_count = tl.sum(cond.to(tl.int64))
    
    tl.atomic_add(sum_ptr, local_sum)
    tl.atomic_add(count_ptr, local_count)
