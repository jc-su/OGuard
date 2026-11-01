import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "quantity_ptr": "*fp64",
    "idx_ptr": "*i64",
    "part_keep_ptr": "*u8",
    "thresh_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr,          # float64 *
    quantity_ptr,       # float64 *
    idx_ptr,            # int64 *
    part_keep_ptr,      # uint8 *
    thresh_ptr,         # float64 *
    out_ptr,            # float64 *
    n_elements,         # i64 scalar
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    # Load part membership flag and part index
    part_keep = tl.load(part_keep_ptr + offs, mask=mask, other=0)
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)

    # Load per‑part threshold only for rows that belong to a filtered part
    thresh = tl.load(thresh_ptr + idx, mask=(part_keep == 1) & mask, other=0.0)

    # Row passes if it belongs to a filtered part and quantity < threshold
    cond = (part_keep == 1) & (quantity < thresh)

    contribution = tl.where(cond, price, 0.0)

    # Block‑wise reduction followed by atomic add to global result
    tl.atomic_add(out_ptr, tl.sum(contribution))
