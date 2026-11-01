import triton
import triton.language as tl

TYPES = {
    "late_ptr": "*i8",
    "supp_ptr": "*i64",
    "distinct_ptr": "*i64",
    "late_distinct_ptr": "*i64",
    "out_ptr": "*i64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 1024}
NUM_WARPS = 4


@triton.jit
def kernel(
    late_ptr,                     # int8*  (0/1)
    supp_ptr,                     # int64*
    distinct_ptr,                 # int64*
    late_distinct_ptr,            # int64*
    out_ptr,                      # int64*   (per‑supplier counters)
    n_elements,                   # total number of rows
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load per‑row data; safe defaults for OOB lanes
    late = tl.load(late_ptr + offs, mask=mask, other=0).to(tl.int1)
    supp = tl.load(supp_ptr + offs, mask=mask, other=-1).to(tl.int64)
    distinct = tl.load(distinct_ptr + offs, mask=mask, other=0).to(tl.int64)
    late_distinct = tl.load(late_distinct_ptr + offs, mask=mask, other=0).to(tl.int64)

    # Predicate logic (identical to reference)
    keep = (
        mask
        & (late == 1)
        & (distinct >= 2)
        & (late_distinct <= 1)
        & (supp >= 0)
    )

    # Safe index for atomic add (when supp==-1 mask==False)
    safe_supp = tl.where(supp >= 0, supp, 0)

    # One atomic add of 1 per qualifying row
    tl.atomic_add(out_ptr + safe_supp, 1, mask=keep)
