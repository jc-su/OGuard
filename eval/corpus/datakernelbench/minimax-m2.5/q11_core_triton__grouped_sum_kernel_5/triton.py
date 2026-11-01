import triton
import triton.language as tl

TYPES = {
    "supplycost_ptr": "*fp64",
    "availqty_ptr": "*fp64",
    "inverse_ptr": "*i64",
    "grouped_ptr": "*fp64",
    "total_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 4


@triton.jit
def kernel(
    supplycost_ptr,
    availqty_ptr,
    inverse_ptr,
    grouped_ptr,
    total_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load supplycost and availqty (float64)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    availqty = tl.load(availqty_ptr + offs, mask=mask, other=0.0)

    # Compute product (value) in registers
    product = supplycost * availqty

    # Add block-level sum to global total
    tl.atomic_add(total_ptr, tl.sum(product))

    # Load group indices (inverse mapping)
    inv = tl.load(inverse_ptr + offs, mask=mask, other=0)

    # Atomically add each row's product to its group accumulator
    tl.atomic_add(grouped_ptr + inv, product)
