import triton
import triton.language as tl

TYPES = {
    "ps_suppkey_ptr": "*i64",
    "ps_availqty_ptr": "*i64",
    "sum_qty_ptr": "*fp64",
    "supp_set_ptr": "*i64",
    "table_mask": "i32",
}

CONSTANTS = {"MAX_PROBE": 64}

NUM_WARPS = 1


@triton.jit
def kernel(
    ps_suppkey_ptr,
    ps_availqty_ptr,
    sum_qty_ptr,
    supp_set_ptr,
    table_mask,
    MAX_PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    supp = tl.load(ps_suppkey_ptr + pid)
    avail = tl.load(ps_availqty_ptr + pid).to(tl.float64)
    qty = tl.load(sum_qty_ptr + pid)
    qual = (qty > 0.0) & (avail > (qty * 0.5))

    x = supp * 1009837
    x = x ^ (x >> 16)
    h = x & table_mask

    done = ~qual
    neg_one = tl.full((), -1, tl.int64)
    neg_two = tl.full((), -2, tl.int64)

    for probe in tl.static_range(0, MAX_PROBE):
        slot = (h + probe) & table_mask
        cmp_val = tl.where(done, neg_two, neg_one)
        old = tl.atomic_cas(supp_set_ptr + slot, cmp_val, supp)
        success = (old == neg_one) & (~done)
        existed = (old == supp) & (~done)
        done = done | success | existed
