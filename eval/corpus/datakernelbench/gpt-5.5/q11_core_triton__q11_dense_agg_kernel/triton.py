import triton
import triton.language as tl

TYPES = {
    "ps_partkey_ptr": "*i64",
    "ps_supplycost_ptr": "*fp64",
    "ps_availqty_ptr": "*fp64",
    "group_values_ptr": "*fp64",
    "group_counts_ptr": "*i32",
    "total_ptr": "*fp64",
    "n_elements": "i32",
    "base_key": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 512}

NUM_WARPS = 4


@triton.jit
def kernel(
    ps_partkey_ptr,
    ps_supplycost_ptr,
    ps_availqty_ptr,
    group_values_ptr,
    group_counts_ptr,
    total_ptr,
    n_elements,
    base_key,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    partkeys = tl.load(ps_partkey_ptr + offs, mask=mask, other=base_key)
    supplycost = tl.load(ps_supplycost_ptr + offs, mask=mask, other=0.0)
    availqty = tl.load(ps_availqty_ptr + offs, mask=mask, other=0.0)

    values = supplycost * availqty
    group_offs = partkeys - base_key

    tl.atomic_add(group_values_ptr + group_offs, values, mask=mask, sem="relaxed")
    tl.atomic_add(
        group_counts_ptr + group_offs,
        tl.full((BLOCK_SIZE,), 1, tl.int32),
        mask=mask,
        sem="relaxed",
    )

    block_sum = tl.sum(tl.where(mask, values, 0.0))
    tl.atomic_add(total_ptr, block_sum, sem="relaxed")
