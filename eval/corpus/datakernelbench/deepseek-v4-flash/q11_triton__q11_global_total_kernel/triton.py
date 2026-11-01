import triton
import triton.language as tl

TYPES = {
    "supplycost_ptr": "*fp64",
    "availqty_ptr": "*fp64",
    "total_out": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
    "num_warps": 4,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    supplycost_ptr, availqty_ptr, total_out, n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    availqty = tl.load(availqty_ptr + offs, mask=mask, other=0.0)
    value = supplycost * availqty
    block_sum = tl.sum(tl.where(mask, value, 0.0))
    tl.atomic_add(total_out, block_sum)
