import triton
import triton.language as tl

TYPES = {
    "cntrycode_ptr": "*i32",
    "acctbal_ptr": "*fp64",
    "outer_mask_ptr": "*i1",
    "count_ptr": "*i64",
    "sum_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    cntrycode_ptr,
    acctbal_ptr,
    outer_mask_ptr,
    count_ptr,
    sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: for each row that passes outer_mask, atomically add
    its balance to the per‑country‑code count and sum."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    cntry = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    outer = tl.load(outer_mask_ptr + offs, mask=mask, other=False)

    keep = outer
    code = cntry

    # atomic add for count (int64)
    count_add = tl.where(keep, 1, 0)
    tl.atomic_add(count_ptr + code, count_add.to(tl.int64))

    # atomic add for sum (float64)
    sum_add = tl.where(keep, acctbal, 0.0)
    tl.atomic_add(sum_ptr + code, sum_add)
