import triton
import triton.language as tl

TYPES = {
    "group_ids_ptr": "*i64",
    "supp_ids_ptr": "*i64",
    "packed_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    group_ids_ptr, supp_ids_ptr, packed_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    g = tl.load(group_ids_ptr + offs, mask=mask, other=0)
    s = tl.load(supp_ids_ptr + offs, mask=mask, other=0)
    
    # Pack into 64-bit int: group_id in high 32 bits, supp_id in low 32 bits
    p = (g << 32) | s
    tl.store(packed_ptr + offs, p, mask=mask)
