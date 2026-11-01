import triton
import triton.language as tl

TYPES = {
    "custkey_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "revenue_ptr": "*fp64",
    "flag_ptr": "*i8",
    "key_min": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    custkey_ptr,
    price_ptr,
    discount_ptr,
    revenue_ptr,
    flag_ptr,
    key_min,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keys = tl.load(custkey_ptr + offs, mask=mask, other=key_min)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    dense_idx = keys - key_min
    revenue = price * (1.0 - discount)

    tl.atomic_add(revenue_ptr + dense_idx, revenue, sem="relaxed", mask=mask)
    tl.store(flag_ptr + dense_idx, 1, mask=mask)
