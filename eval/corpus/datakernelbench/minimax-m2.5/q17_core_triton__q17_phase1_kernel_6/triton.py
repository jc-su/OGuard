import triton
import triton.language as tl

TYPES = {
    "part_idx_ptr": "*i64",
    "mask_ptr": "*u8",
    "qty_ptr": "*fp64",
    "sum_qty_ptr": "*fp64",
    "cnt_qty_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    part_idx_ptr,      # int64 per‑row index into sorted part keys
    mask_ptr,          # uint8 mask indicating part‑membership
    qty_ptr,           # float64 quantity
    sum_qty_ptr,       # float64 output: sum of quantities per part
    cnt_qty_ptr,       # float64 output: count of rows per part
    n_elements: int,   # number of lineitem rows
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load membership mask, part index, and quantity
    mask_vals = tl.load(mask_ptr + offs, mask=mask, other=0).to(tl.int1)
    part_idx = tl.load(part_idx_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)

    # Compute contributions only for rows that belong to filtered parts
    # Use a loop to do per-element atomic adds (simple and correct)
    for i in range(BLOCK_SIZE):
        global_off = pid * BLOCK_SIZE + i
        if global_off < n_elements:
            m = tl.load(mask_ptr + global_off).to(tl.int1)
            if m:
                p = tl.load(part_idx_ptr + global_off)
                q = tl.load(qty_ptr + global_off)
                tl.atomic_add(sum_qty_ptr + p, q)
                tl.atomic_add(cnt_qty_ptr + p, 1.0)
