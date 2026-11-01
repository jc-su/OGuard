import triton
import triton.language as tl

TYPES = {
    "l_suppkey_ptr": "*i64",
    "l_price_ptr": "*fp64",
    "l_disc_ptr": "*fp64",
    "l_shipdate_ptr": "*i32",
    "suppkey_map_ptr": "*i32",
    "shipdate_lo": "i32",
    "shipdate_hi": "i32",
    "totals_ptr": "*fp64",
    "present_ptr": "*i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_suppkey_ptr, l_price_ptr, l_disc_ptr, l_shipdate_ptr,
    suppkey_map_ptr,  # Maps suppkey to dense index
    shipdate_lo, shipdate_hi,
    totals_ptr, present_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load and filter by date
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=shipdate_hi)
    keep = (shipdate >= shipdate_lo) & (shipdate < shipdate_hi) & mask

    # Calculate revenue and get suppkey
    price = tl.load(l_price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(l_disc_ptr + offs, mask=keep, other=0.0)
    revenue = price * (1.0 - disc)
    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)

    # Get dense index from mapping
    dense_idx = tl.load(suppkey_map_ptr + suppkey, mask=keep, other=-1)

    # Atomic operations using dense index
    tl.atomic_add(totals_ptr + dense_idx, revenue, mask=keep & (dense_idx >= 0))
    tl.atomic_add(present_ptr + dense_idx, 1, mask=keep & (dense_idx >= 0))
