import triton
import triton.language as tl

TYPES = {
    "custkey_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "min_key": "i32",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    custkey_ptr,
    price_ptr,
    discount_ptr,
    out_ptr,
    min_key,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: compute revenue = price * (1 - discount) and atomic‑add into per‑customer bucket."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input data (use 0 for out‑of‑range lanes)
    cust = tl.load(custkey_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Revenue per row
    revenue = price * (1.0 - discount)

    # Compute bucket index: key - min_key (valid only for lanes that are kept)
    idx = tl.where(mask, cust - min_key, 0)

    # Atomic add the revenue into the corresponding bucket.
    # Masked‑off lanes add zero, which is harmless.
    tl.atomic_add(out_ptr + idx, revenue)
