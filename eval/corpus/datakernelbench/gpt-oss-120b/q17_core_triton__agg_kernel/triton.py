import triton
import triton.language as tl

TYPES = {
    "idx_ptr": "*i64",
    "qty_ptr": "*fp64",
    "sum_ptr": "*fp64",
    "cnt_ptr": "*fp64",
    "n_rows": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 1024}
NUM_WARPS = 4


@triton.jit
def kernel(
    idx_ptr,                # int64*  (part index for each selected row)
    qty_ptr,                # fp64*   (quantity for each selected row)
    sum_ptr,                # fp64*   (per‑part sum of quantity)
    cnt_ptr,                # fp64*   (per‑part count, stored as fp64)
    n_rows,                 # int64   (number of selected rows)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)

    # Guard against out‑of‑bounds contributions
    add_qty = tl.where(mask, qty, 0.0)
    add_cnt = tl.where(mask, 1.0, 0.0)

    tl.atomic_add(sum_ptr + idx, add_qty)   # fp64 sum
    tl.atomic_add(cnt_ptr + idx, add_cnt)   # fp64 count
