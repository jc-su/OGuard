import triton
import triton.language as tl

TYPES = {
    "idx_ptr": "*i64",
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "thresh_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_rows": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 4


@triton.jit
def kernel(
    idx_ptr,                # int64*  (part index for each selected row)
    qty_ptr,                # fp64*   (quantity for each selected row)
    price_ptr,              # fp64*   (extendedprice for each selected row)
    thresh_ptr,             # fp64*   (per‑part 0.2 * avg_quantity)
    out_ptr,                # fp64*   (global accumulator)
    n_rows,                 # int64   (number of selected rows)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)

    thresh = tl.load(thresh_ptr + idx, mask=mask, other=0.0)

    keep = mask & (qty < thresh)

    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, price, 0.0)))
