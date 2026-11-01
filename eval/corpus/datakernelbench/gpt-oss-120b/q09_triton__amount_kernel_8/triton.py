import triton
import triton.language as tl

TYPES = {
    "ext_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "sup_ptr": "*fp64",
    "qty_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 8


@triton.jit
def kernel(
    ext_ptr,          # float64 *
    disc_ptr,         # float64 *
    sup_ptr,          # float64 *
    qty_ptr,          # float64 *
    out_ptr,          # float64 *
    n_elements,       # i64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load each column; OOB lanes receive a benign value
    ext = tl.load(ext_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    sup = tl.load(sup_ptr + offs, mask=mask, other=0.0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)

    # amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
    amt = ext * (1.0 - disc) - sup * qty

    tl.store(out_ptr + offs, amt, mask=mask)
