import triton
import triton.language as tl

TYPES = {
    "priority_ptr": "*i32",
    "filter_mask_ptr": "*i32",
    "counts_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 4


@triton.jit
def kernel(
    priority_ptr,
    filter_mask_ptr,
    counts_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q4 priority counting with block-level reduction.
    
    Each block accumulates local counts for all 5 priorities using tl.sum,
    then does ONE atomic add per priority. This reduces atomic contention
    from O(n) to O(num_blocks).
    
    Optimized for H100: large blocks, minimal atomics, coalesced access.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load priority codes and filter mask
    priority = tl.load(priority_ptr + offs, mask=mask, other=0)
    keep = tl.load(filter_mask_ptr + offs, mask=mask, other=0)
    
    # Block-level reduction: count per priority in this block
    # tl.sum reduces across the BLOCK_SIZE dimension efficiently
    count_0 = tl.sum(tl.where((priority == 0) & (keep == 1), 1, 0).to(tl.int64))
    count_1 = tl.sum(tl.where((priority == 1) & (keep == 1), 1, 0).to(tl.int64))
    count_2 = tl.sum(tl.where((priority == 2) & (keep == 1), 1, 0).to(tl.int64))
    count_3 = tl.sum(tl.where((priority == 3) & (keep == 1), 1, 0).to(tl.int64))
    count_4 = tl.sum(tl.where((priority == 4) & (keep == 1), 1, 0).to(tl.int64))
    
    # One atomic add per priority per block (5 atomics per block)
    tl.atomic_add(counts_ptr + 0, count_0, sem="relaxed")
    tl.atomic_add(counts_ptr + 1, count_1, sem="relaxed")
    tl.atomic_add(counts_ptr + 2, count_2, sem="relaxed")
    tl.atomic_add(counts_ptr + 3, count_3, sem="relaxed")
    tl.atomic_add(counts_ptr + 4, count_4, sem="relaxed")
