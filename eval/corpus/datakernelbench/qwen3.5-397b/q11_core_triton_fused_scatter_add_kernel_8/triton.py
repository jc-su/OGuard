import triton
import triton.language as tl

TYPES = {
    "cost_ptr": "*fp64",
    "qty_ptr": "*fp64",
    "inverse_ptr": "*i64",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 8


@triton.jit
def kernel(
    cost_ptr, qty_ptr, inverse_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load inputs with mask to handle OOB safely
    c = tl.load(cost_ptr + offs, mask=mask, other=0.0)
    q = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    inv = tl.load(inverse_ptr + offs, mask=mask, other=0)
    
    # Compute value per row
    val = c * q
    
    # Scatter-add into group_sums buffer using inverse indices
    # mask ensures OOB threads do not perform atomic_add
    tl.atomic_add(out_ptr + inv, val, mask=mask)
