import triton
import triton.language as tl

TYPES = {
    "ps_partkey_ptr": "*i64",
    "ps_suppkey_ptr": "*i64",
    "ht_part_ptr": "*i64",
    "ht_supp_ptr": "*i64",
    "ht_idx_ptr": "*i64",
    "table_mask": "i32",
}

CONSTANTS = {"MAX_PROBE": 64}

NUM_WARPS = 1


@triton.jit
def kernel(
    ps_partkey_ptr,
    ps_suppkey_ptr,
    ht_part_ptr,
    ht_supp_ptr,
    ht_idx_ptr,
    table_mask,
    MAX_PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    p = tl.load(ps_partkey_ptr + pid)
    s = tl.load(ps_suppkey_ptr + pid)

    x = (p * 1000003) ^ (s * 1009837)
    x = x ^ (x >> 16)
    h = x & table_mask

    done = False
    neg_one = tl.full((), -1, tl.int64)
    neg_two = tl.full((), -2, tl.int64)
    pid64 = pid.to(tl.int64)

    for probe in tl.static_range(0, MAX_PROBE):
        slot = (h + probe) & table_mask
        cmp_val = tl.where(done, neg_two, neg_one)
        old = tl.atomic_cas(ht_idx_ptr + slot, cmp_val, pid64)
        success = (old == neg_one) & (~done)
        tl.store(ht_part_ptr + slot, p, mask=success)
        tl.store(ht_supp_ptr + slot, s, mask=success)
        done = done | success
