import triton
import triton.language as tl

TYPES = {
    "nation_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 8


@triton.jit
def kernel(
    nation_ptr: tl.pointer_type(tl.int64),      # int64*
    price_ptr: tl.pointer_type(tl.float64),    # fp64*
    discount_ptr: tl.pointer_type(tl.float64), # fp64*
    out_ptr: tl.pointer_type(tl.float64),      # fp64* (size = num_nations)
    n_elements,                                 # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns (masked)
    nation = tl.load(nation_ptr + offs, mask=mask, other=0)
    price  = tl.load(price_ptr  + offs, mask=mask, other=0.0)
    disc   = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Compute revenue = price * (1 - discount)
    revenue = price * (1.0 - disc)

    # Atomic add per nation; masked lanes add zero
    tl.atomic_add(out_ptr + nation, tl.where(mask, revenue, 0.0))
