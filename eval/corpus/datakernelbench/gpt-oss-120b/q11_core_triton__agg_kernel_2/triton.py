import triton
import triton.language as tl

TYPES = {
    "supply_ptr": "*fp64",
    "avail_ptr": "*fp64",
    "out_ptr": "*fp64",
    "total_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 1024}
NUM_WARPS = 8


@triton.jit
def kernel(
    supply_ptr,          # float64 *
    avail_ptr,           # float64 *
    out_ptr,             # float64 *
    total_ptr,           # float64 *
    n_elements,          # i64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Valid‑index mask
    mask = offs < n_elements

    # Load inputs; out‑of‑bounds lanes receive 0.0
    supply = tl.load(supply_ptr + offs, mask=mask, other=0.0)
    avail  = tl.load(avail_ptr  + offs, mask=mask, other=0.0)

    # Per‑row product
    val = supply * avail

    # Write product to output buffer
    tl.store(out_ptr + offs, val, mask=mask)

    # Zero OOB lanes before reduction
    val_masked = tl.where(mask, val, 0.0)

    # Block‑wise sum then atomic add to global total
    block_sum = tl.sum(val_masked)
    tl.atomic_add(total_ptr, block_sum)
