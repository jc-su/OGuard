import triton
import triton.language as tl

TYPES = {
    "custkey_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "revenue_ptr": "*fp64",
    "n_rows": "i32",
    "n_revenue": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    custkey_ptr,
    price_ptr,
    discount_ptr,
    revenue_ptr,
    n_rows,
    n_revenue,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    custkey = tl.load(custkey_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    valid = mask & (custkey >= 0) & (custkey < n_revenue)
    revenue = price * (1.0 - discount)

    tl.atomic_add(revenue_ptr + custkey, revenue, sem="relaxed", mask=valid)
