import triton
import triton.language as tl

TYPES = {
    "volume_ptr": "*fp64",
    "is_brazil_ptr": "*i1",
    "inv_ptr": "*i64",
    "brazil_ptr": "*fp64",
    "total_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 2048
}

NUM_WARPS = 8


@triton.jit
def kernel(
    volume_ptr, is_brazil_ptr, inv_ptr,
    brazil_ptr, total_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for market share aggregation.
    
    Computes per-group sums:
    - brazil_vol[group] = sum(volume where is_brazil)
    - total_vol[group] = sum(volume)
    
    Uses atomic_add for thread-safe accumulation across blocks.
    Memory-bound optimization: single pass through data, coalesced access.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load data with bounds checking
    volume = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    is_brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=0)
    group_idx = tl.load(inv_ptr + offs, mask=mask, other=0)
    
    # Compute Brazil volume (conditional in registers, no extra HBM)
    brazil_vol = tl.where(is_brazil, volume, 0.0)
    
    # Atomic accumulation into per-group bins
    # mask ensures OOB lanes don't participate in atomics
    tl.atomic_add(brazil_ptr + group_idx, brazil_vol, mask=mask)
    tl.atomic_add(total_ptr + group_idx, volume, mask=mask)
