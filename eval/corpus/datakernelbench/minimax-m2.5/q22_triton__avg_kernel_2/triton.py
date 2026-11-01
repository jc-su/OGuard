import triton
import triton.language as tl

TYPES = {
    "acctbal_ptr": "*fp64",
    "cntrycode_ptr": "*i32",
    "in_mask_ptr": "*i32",
    "sum_ptr": "*fp64",
    "count_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    acctbal_ptr, cntrycode_ptr, in_mask_ptr, 
    sum_ptr, count_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load predicate mask (whether cntrycode is in target set)
    in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    
    # Load cntrycode - use as sentinel if not in set
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    
    # Load acctbal - use 0.0 as sentinel if predicate fails
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    
    # Predicate: in_mask AND acctbal > 0.0
    keep = (in_mask != 0) & (acctbal > 0.0)
    
    # Sum and count where predicate holds
    sum_val = tl.sum(tl.where(keep, acctbal, 0.0))
    count_val = tl.sum(tl.where(keep, 1.0, 0))
    
    tl.atomic_add(sum_ptr, sum_val)
    tl.atomic_add(count_ptr, count_val)
