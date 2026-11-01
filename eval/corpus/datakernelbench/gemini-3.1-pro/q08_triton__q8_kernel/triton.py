import triton
import triton.language as tl

TYPES = {
    "year_ptr": "*i32",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "is_target_ptr": "*i32",
    "sum_target_ptr": "*fp64",
    "sum_total_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 4


@triton.jit
def kernel(
    year_ptr, price_ptr, discount_ptr, is_target_ptr,
    sum_target_ptr, sum_total_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    year = tl.load(year_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    is_target = tl.load(is_target_ptr + offs, mask=mask, other=0)

    volume = price * (1.0 - discount)
    target_volume = tl.where(is_target > 0, volume, 0.0)

    # Use year directly as index to sum arrays
    valid_idx = (year >= 1900) & (year < 2500) & mask

    tl.atomic_add(sum_total_ptr + year, volume, mask=valid_idx)
    tl.atomic_add(sum_target_ptr + year, target_volume, mask=valid_idx)
