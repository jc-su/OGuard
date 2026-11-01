import triton
import triton.language as tl

TYPES = {
    "line_key_ptr": "*i64",
    "line_commit_ptr": "*i32",
    "line_receipt_ptr": "*i32",
    "out_key_ptr": "*i64",
    "out_valid_ptr": "*i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 8


@triton.jit
def kernel(
    line_key_ptr,
    line_commit_ptr,
    line_receipt_ptr,
    out_key_ptr,
    out_valid_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Filter lineitems where commit < receipt using Triton.
    Fuses comparison with key extraction in one kernel pass.
    Memory-bound optimization: single HBM pass for filter+project.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    l_commit = tl.load(line_commit_ptr + offs, mask=mask, other=0)
    l_receipt = tl.load(line_receipt_ptr + offs, mask=mask, other=0)
    keep = mask & (l_commit < l_receipt)
    
    line_key = tl.load(line_key_ptr + offs, mask=keep, other=0)
    tl.store(out_key_ptr + offs, line_key, mask=keep)
    tl.store(out_valid_ptr + offs, keep.to(tl.int32), mask=mask)
