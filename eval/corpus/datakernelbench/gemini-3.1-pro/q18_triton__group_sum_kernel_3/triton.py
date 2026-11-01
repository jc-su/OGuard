import triton
import triton.language as tl

TYPES = {
    "orderkey_ptr": "*i64",
    "quantity_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    orderkey_ptr, quantity_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Scans the lineitem quantities and directly accumulates them into a global
    array indexed by orderkey using fast atomic additions.
    This replaces a multi-pass sort/unique/scatter_add with a single pass,
    significantly reducing memory bandwidth requirements and temporary allocations.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load orderkeys and quantities
    okey = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    
    # Hopper L2 atomics are extremely fast and sufficient for this 1D histogram pattern.
    tl.atomic_add(out_ptr + okey, qty, mask=mask)
