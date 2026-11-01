import triton
import triton.language as tl

TYPES = {
    "supp_id_ptr": "*i32",
    "late_ptr": "*i8",
    "distinct_ptr": "*i32",
    "late_dist_ptr": "*i32",
    "out_ptr": "*i64",
    "n_rows": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 4


@triton.jit
def kernel(
    supp_id_ptr,    # int32 *
    late_ptr,       # int8  *
    distinct_ptr,   # int32 *
    late_dist_ptr,  # int32 *
    out_ptr,        # int64 *  (per‑supplier counters)
    n_rows,         # i64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    # Load columns
    supp = tl.load(supp_id_ptr + offs, mask=mask, other=-1)
    late = tl.load(late_ptr + offs, mask=mask, other=0).to(tl.int1)
    distinct = tl.load(distinct_ptr + offs, mask=mask, other=0)
    late_dist = tl.load(late_dist_ptr + offs, mask=mask, other=0)

    # Predicate composition
    keep = (
        mask
        & (late != 0)
        & (distinct >= 2)
        & (late_dist <= 1)
        & (supp >= 0)
    )

    # Atomic increment of the per‑supplier counter.
    # out_ptr + supp is a block pointer; atomic_add works lane‑wise.
    tl.atomic_add(out_ptr + supp, 1, mask=keep)
