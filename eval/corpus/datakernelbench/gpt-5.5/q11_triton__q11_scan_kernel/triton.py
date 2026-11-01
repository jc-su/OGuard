import triton
import triton.language as tl

TYPES = {
    "ps_partkey_ptr": "*i64",
    "ps_suppkey_ptr": "*i64",
    "ps_supplycost_ptr": "*fp64",
    "ps_availqty_ptr": "*i64",
    "supplier_mask_ptr": "*i8",
    "group_values_ptr": "*fp64",
    "total_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"supplier_mask_n": 100001, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    ps_partkey_ptr,
    ps_suppkey_ptr,
    ps_supplycost_ptr,
    ps_availqty_ptr,
    supplier_mask_ptr,
    group_values_ptr,
    total_ptr,
    supplier_mask_n: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_elements

    suppkey = tl.load(ps_suppkey_ptr + offs, mask=row_mask, other=-1)
    supp_in_range = row_mask & (suppkey >= 0) & (suppkey < supplier_mask_n)
    supp_keep = tl.load(supplier_mask_ptr + suppkey, mask=supp_in_range, other=0) != 0
    keep = row_mask & supp_keep

    partkey = tl.load(ps_partkey_ptr + offs, mask=keep, other=0)
    supplycost = tl.load(ps_supplycost_ptr + offs, mask=keep, other=0.0)
    availqty_i64 = tl.load(ps_availqty_ptr + offs, mask=keep, other=0)
    value = supplycost * availqty_i64.to(tl.float64)

    block_total = tl.sum(tl.where(keep, value, 0.0))
    tl.atomic_add(total_ptr, block_total, sem="relaxed")

    tl.atomic_add(group_values_ptr + partkey, value, sem="relaxed", mask=keep)
