import triton
import triton.language as tl

TYPES = {
    "l_orderkey": "*i32",
    "l_suppkey": "*i32",
    "l_receiptdate": "*i32",
    "l_commitdate": "*i32",
    "o_supp_min": "*i32",
    "o_supp_max": "*i32",
    "o_late_min": "*i32",
    "o_late_max": "*i32",
    "target_supp_map": "*i32",
    "order_status_F": "*u8",
    "counts": "*i32",
    "num_lineitems": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_orderkey, l_suppkey, l_receiptdate, l_commitdate,
    o_supp_min, o_supp_max, o_late_min, o_late_max,
    target_supp_map, order_status_F, counts,
    num_lineitems,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_lineitems

    o_key = tl.load(l_orderkey + offs, mask=mask, other=0)
    s_key = tl.load(l_suppkey + offs, mask=mask, other=0)
    receipt = tl.load(l_receiptdate + offs, mask=mask, other=0)
    commit = tl.load(l_commitdate + offs, mask=mask, other=1)

    # Base predicate: l1.l_receiptdate > l1.l_commitdate
    late = receipt > commit
    keep = mask & late

    # Filter target supplier (also resolves index for counts array)
    supp_id = tl.load(target_supp_map + s_key, mask=keep, other=-1)
    keep = keep & (supp_id >= 0)

    # Filter order status 'F'
    is_F = tl.load(order_status_F + o_key, mask=keep, other=0)
    keep = keep & (is_F == 1)

    # EXISTS (another supplier for the same order) -> min < max implies >= 2 distinct suppliers
    s_min = tl.load(o_supp_min + o_key, mask=keep, other=0)
    s_max = tl.load(o_supp_max + o_key, mask=keep, other=0)
    keep = keep & (s_min < s_max)

    # NOT EXISTS (another late supplier for the same order) -> min == max implies exactly 1 distinct late supplier
    # Since l1 is late itself, it means the only late supplier is l1's supplier.
    l_min = tl.load(o_late_min + o_key, mask=keep, other=0)
    l_max = tl.load(o_late_max + o_key, mask=keep, other=1)
    keep = keep & (l_min == l_max)

    # Increment count for the valid supplier
    supp_id_valid = tl.where(keep, supp_id, 0)
    tl.atomic_add(counts + supp_id_valid, 1, mask=keep)
