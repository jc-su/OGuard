import triton
import triton.language as tl

TYPES = {
    "supplycost_ptr": "*fp64",
    "availqty_ptr": "*fp64",
    "inverse_ptr": "*i64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
    "n_groups": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 2048
}

NUM_WARPS = 4


@triton.jit
def kernel(
    supplycost_ptr, availqty_ptr, inverse_ptr,
    out_ptr, n_elements, n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q11: compute value = supplycost * availqty and
    scatter_add into grouped output by inverse indices.
    
    Memory-bound optimization: single pass through data, computes value in registers,
    atomic_add directly to grouped output. Avoids separate multiply + scatter passes.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load data with safe defaults for OOB lanes
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    availqty = tl.load(availqty_ptr + offs, mask=mask, other=0.0)
    inverse = tl.load(inverse_ptr + offs, mask=mask, other=0)
    
    # Compute value in registers (no extra HBM traffic)
    value = supplycost * availqty
    
    # Atomic add to grouped output at inverse[group_idx]
    tl.atomic_add(out_ptr + inverse, value, mask=mask)
