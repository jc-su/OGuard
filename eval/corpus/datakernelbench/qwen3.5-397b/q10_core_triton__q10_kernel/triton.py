import triton
import triton.language as tl

TYPES = {
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "inverse_ptr": "*i64",
    "revenue_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 1024
}

NUM_WARPS = 4


@triton.jit
def kernel(
    extendedprice_ptr,
    discount_ptr,
    inverse_ptr,
    revenue_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: compute revenue = extendedprice * (1 - discount) and scatter-add to output.
    Memory-bound optimization: single pass through input, no intermediate tensor allocation.
    Uses atomic_add for group-by aggregation via inverse indices from torch.unique.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load inputs with bounds checking
    extendedprice = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    inverse = tl.load(inverse_ptr + offs, mask=mask, other=0)
    
    # Compute revenue per row: extendedprice * (1 - discount)
    revenue = extendedprice * (1.0 - discount)
    
    # Scatter-add to output revenue array using inverse indices for group-by
    tl.atomic_add(revenue_ptr + inverse, revenue, mask=mask)
