import triton
import triton.language as tl

TYPES = {
    "acctbal_ptr": "*fp64",
    "cntrycode_ptr": "*i32",
    "in_mask_ptr": "*i1",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 1024
}

NUM_WARPS = 8


@triton.jit
def kernel(
    acctbal_ptr,
    cntrycode_ptr,
    in_mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute sum and count of c_acctbal where cntrycode in set AND c_acctbal > 0.
    Output: out_ptr[0] = sum, out_ptr[1] = count (both float64)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load cntrycode and in_mask
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_set = tl.load(in_mask_ptr + offs, mask=mask, other=False)
    
    # Load acctbal with safe other
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    
    # Filter: in_set AND acctbal > 0
    keep = mask & in_set & (acctbal > 0.0)
    
    # Conditional load for acctbal (only load where keep is true)
    acctbal_filtered = tl.load(acctbal_ptr + offs, mask=keep, other=0.0)
    
    # Accumulate sum and count
    block_sum = tl.sum(tl.where(keep, acctbal_filtered, 0.0))
    block_count = tl.sum(tl.where(keep, 1.0, 0.0))
    
    # Atomic add to global accumulators
    tl.atomic_add(out_ptr, block_sum)
    tl.atomic_add(out_ptr + 1, block_count)
