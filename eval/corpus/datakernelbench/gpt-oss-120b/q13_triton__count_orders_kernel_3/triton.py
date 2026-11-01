import triton
import triton.language as tl

TYPES = {
    "o_custkey_ptr": "*i64",
    "mask_ptr": "*u8",
    "out_ptr": "*i32",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 4


@triton.jit
def kernel(
    o_custkey_ptr,           # int64*
    mask_ptr,                # uint8*  (1 = keep)
    out_ptr,                 # int32*  (size = num_customers+1)
    n_elements,              # total number of rows in orders
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    # Load columns (masked loads keep out‑of‑bounds safe)
    custkey = tl.load(o_custkey_ptr + offs, mask=valid, other=0)
    keep    = tl.load(mask_ptr + offs,    mask=valid, other=0)

    # Customer keys start at 1 → cast to int32 index
    idx = tl.cast(custkey, tl.int32)

    # Convert keep (uint8) to int32 0/1 for atomic add
    inc = tl.cast(keep, tl.int32)

    tl.atomic_add(out_ptr + idx, inc)
