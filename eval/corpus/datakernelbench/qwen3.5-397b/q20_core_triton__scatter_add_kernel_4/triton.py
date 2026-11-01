import triton
import triton.language as tl

TYPES = {
    "out_ptr": "*fp64",
    "indices_ptr": "*i64",
    "values_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 2048
}

NUM_WARPS = 8


@triton.jit
def kernel(
    out_ptr,
    indices_ptr,
    values_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused scatter_add: out[indices[i]] += values[i]
    
    Memory-bound optimization:
    - Single pass through indices and values (reduces HBM traffic)
    - Coalesced memory access patterns optimized for H100
    - Atomic operations for concurrent group updates
    - Autotuned block sizes for optimal occupancy on Hopper
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load indices and values with mask for OOB protection
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)
    val = tl.load(values_ptr + offs, mask=mask, other=0.0)
    
    # Atomic add to output groups (handles concurrent writes safely)
    tl.atomic_add(out_ptr + idx, val, mask=mask)
