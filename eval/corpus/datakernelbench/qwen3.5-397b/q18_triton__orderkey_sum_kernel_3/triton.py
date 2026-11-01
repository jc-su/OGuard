import triton
import triton.language as tl

TYPES = {
    "quantity_ptr": "*fp64",
    "inverse_ptr": "*i64",
    "sum_ptr": "*fp64",
    "n_elements": "i32",
    "n_unique": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 2048
}

NUM_WARPS = 32


@triton.jit
def kernel(
    quantity_ptr,
    inverse_ptr,
    sum_ptr,
    n_elements,
    n_unique,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes per-orderkey sum of quantity using inverse indices from torch.unique.
    Optimized for H100 with aggressive block sizes, warps, and pipeline stages.
    Uses vectorized loads where possible for better memory throughput.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offs) < n_elements
    
    # Load inverse indices and quantities with vectorized access
    group_idx = tl.load(inverse_ptr + block_start + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + block_start + offs, mask=mask, other=0.0)
    
    # Atomic add to group sums - H100 has good atomic throughput
    tl.atomic_add(sum_ptr + group_idx, qty, mask=mask)
