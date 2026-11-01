import triton
import triton.language as tl

TYPES = {
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "inverse_ptr": "*i64",
    "agg_revenue_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 2048
}

NUM_WARPS = 8


@triton.jit
def kernel(
    extendedprice_ptr,
    discount_ptr,
    inverse_ptr,
    agg_revenue_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q10 revenue aggregation.
    
    Computes revenue = extendedprice * (1 - discount) per row,
    then atomically accumulates into per-customer buckets using inverse indices.
    
    Memory-bound optimization: single pass through data, revenue calc + agg fused,
    avoiding intermediate tensor materialization from reference torch.scatter_add_.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load extendedprice and discount (fp64 for precision)
    ext_price = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    
    # Compute revenue in registers: l_extendedprice * (1 - l_discount)
    revenue = ext_price * (1.0 - discount)
    
    # Load inverse indices (mapping each row to unique customer bucket 0..num_unique-1)
    inverse = tl.load(inverse_ptr + offs, mask=mask, other=0)
    
    # Atomic add to per-customer aggregation buffer
    # Each thread block processes BLOCK_SIZE rows, atomically accumulating
    tl.atomic_add(agg_revenue_ptr + inverse, revenue, mask=mask)
