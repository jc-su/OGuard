import triton
import triton.language as tl

TYPES = {
    "ps_suppkey_ptr": "*i64",
    "ps_partkey_ptr": "*i64",
    "ps_supplycost_ptr": "*fp64",
    "ps_availqty_ptr": "*fp64",
    "is_nation_supp_ptr": "*i32",
    "out_sums_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    ps_suppkey_ptr,
    ps_partkey_ptr,
    ps_supplycost_ptr,
    ps_availqty_ptr,
    is_nation_supp_ptr,
    out_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load supplier key and check validity (is it in the target nation?)
    suppkey = tl.load(ps_suppkey_ptr + offs, mask=mask, other=0)
    valid = tl.load(is_nation_supp_ptr + suppkey, mask=mask, other=0)
    keep = mask & (valid == 1)

    # Conditionally load heavy columns (low selectivity on valid suppliers)
    partkey = tl.load(ps_partkey_ptr + offs, mask=keep, other=0)
    cost = tl.load(ps_supplycost_ptr + offs, mask=keep, other=0.0)
    qty = tl.load(ps_availqty_ptr + offs, mask=keep, other=0.0)

    val = cost * qty

    # Atomic add to the group's slot in out_sums array
    tl.atomic_add(out_sums_ptr + partkey, val, mask=keep)
