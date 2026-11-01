import triton
import triton.language as tl

TYPES = {
    "l_partkey_ptr": "*i64",
    "l_quantity_ptr": "*fp64",
    "is_valid_ptr": "*u8",
    "sum_qty_ptr": "*fp64",
    "cnt_qty_ptr": "*fp64",
    "max_pk": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_partkey_ptr, l_quantity_ptr,
    is_valid_ptr, sum_qty_ptr, cnt_qty_ptr,
    max_pk, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load partkey, default to 0 for out-of-bounds lanes
    pk = tl.load(l_partkey_ptr + offs, mask=mask, other=0)
    
    # Ensure safe indexing into the dense lookup tables
    valid_idx = (pk >= 0) & (pk <= max_pk)
    safe_pk = tl.where(valid_idx, pk, 0)
    
    # Check if this partkey is one of the filtered parts
    is_valid = tl.load(is_valid_ptr + safe_pk, mask=mask & valid_idx, other=0)
    keep = mask & valid_idx & (is_valid == 1)
    
    # Load quantity only for valid items to save bandwidth
    qty = tl.load(l_quantity_ptr + offs, mask=keep, other=0.0).to(tl.float64)
    one = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float64)
    
    # Atomically accumulate sum and count directly to HBM
    tl.atomic_add(sum_qty_ptr + safe_pk, qty, mask=keep)
    tl.atomic_add(cnt_qty_ptr + safe_pk, one, mask=keep)
