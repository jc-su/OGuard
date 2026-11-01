import triton
import triton.language as tl

TYPES = {
    "priority_ptr": "*i32",
    "n_elements": "i32",
    "out_ptr": "*i64",
}
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 4


@triton.jit
def kernel(
    priority_ptr,          # *int32  (order priority codes)
    n_elements,           # i64    (number of elements)
    out_ptr,              # *int64 (per‑priority counters)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load priority code (int32) and promote to int64 for pointer arithmetic
    pr = tl.load(priority_ptr + offs, mask=mask, other=0)
    pr = pr.to(tl.int64)

    # Increment by 1 where the lane is active, otherwise add 0 (no‑op)
    inc = tl.where(mask, 1, 0)
    tl.atomic_add(out_ptr + pr, inc)
