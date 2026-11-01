import triton
import triton.language as tl

TYPES = {
    "partkey_ptr": "*i64",
    "supplycost_ptr": "*fp64",
    "availqty_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    partkey_ptr, supplycost_ptr, availqty_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load inputs
    k = tl.load(partkey_ptr + offs, mask=mask, other=0)
    cost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    qty = tl.load(availqty_ptr + offs, mask=mask, other=0.0)
    
    # Project
    val = cost * qty
    
    # Scatter aggregate directly into the global array
    tl.atomic_add(out_ptr + k, val, mask=mask)
