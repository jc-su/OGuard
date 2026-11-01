import triton
import triton.language as tl

TYPES = {
    "custkey_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "revenue_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    custkey_ptr,
    price_ptr,
    discount_ptr,
    revenue_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load columns – use a safe "other" so masked‑out lanes get zeros
    custkey = tl.load(custkey_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # compute revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)

    # atomic add to per‑customer bucket – masked lanes write zero
    tl.atomic_add(revenue_ptr + custkey, revenue)
