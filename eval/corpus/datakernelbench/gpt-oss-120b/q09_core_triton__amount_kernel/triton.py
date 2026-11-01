import triton
import triton.language as tl

TYPES = {
    "out_ptr": "*fp64",
    "a_ptr": "*fp64",
    "d_ptr": "*fp64",
    "c_ptr": "*fp64",
    "q_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    out_ptr,            # pointer to output amount column
    a_ptr,              # l_extendedprice
    d_ptr,              # l_discount
    c_ptr,              # ps_supplycost
    q_ptr,              # l_quantity
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load inputs (fp64)
    a = tl.load(a_ptr + offs, mask=mask)
    d = tl.load(d_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    q = tl.load(q_ptr + offs, mask=mask)

    # amount = a * (1 - d) - c * q   (all fp64)
    amount = a * (1.0 - d) - c * q

    # Store result
    tl.store(out_ptr + offs, amount, mask=mask)
