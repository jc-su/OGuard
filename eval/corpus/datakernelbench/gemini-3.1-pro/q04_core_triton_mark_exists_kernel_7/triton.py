import triton
import triton.language as tl

TYPES = {
    "l_okey_ptr": "*i64",
    "l_commit_ptr": "*i32",
    "l_receipt_ptr": "*i32",
    "exists_map_ptr": "*i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_okey_ptr, l_commit_ptr, l_receipt_ptr, exists_map_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    commit = tl.load(l_commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(l_receipt_ptr + offs, mask=mask, other=0)
    
    keep = mask & (commit < receipt)
    okey = tl.load(l_okey_ptr + offs, mask=keep, other=0)
    
    # Store 1 in the map for qualifying orderkeys
    val = tl.full([BLOCK_SIZE], 1, dtype=tl.int32)
    tl.store(exists_map_ptr + okey, val, mask=keep)
