import triton
import triton.language as tl

TYPES = {
    "part_idx_ptr": "*i64",
    "mask_ptr": "*u8",
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "threshold_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    part_idx_ptr,      # int64 per‑row index into sorted part keys
    mask_ptr,          # uint8 mask indicating part‑membership
    qty_ptr,           # float64 quantity
    price_ptr,         # float64 extended price
    threshold_ptr,     # float64 per‑part threshold (0.2 * avg qty)
    out_ptr,           # float64 output accumulator
    n_elements: int,   # number of lineitem rows
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Process each element individually with atomic adds
    for i in range(BLOCK_SIZE):
        global_off = pid * BLOCK_SIZE + i
        if global_off < n_elements:
            mask_val = tl.load(mask_ptr + global_off).to(tl.int1)
            if mask_val:
                p = tl.load(part_idx_ptr + global_off)
                q = tl.load(qty_ptr + global_off)
                price = tl.load(price_ptr + global_off)
                thresh = tl.load(threshold_ptr + p)
                if q < thresh:
                    tl.atomic_add(out_ptr, price)
