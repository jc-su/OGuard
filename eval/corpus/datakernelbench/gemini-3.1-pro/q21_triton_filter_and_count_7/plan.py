import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["num_lineitems"],
)
@triton.jit
def build_order_stats(
    l_orderkey, l_suppkey, l_receiptdate, l_commitdate,
    order_status_F,
    o_supp_min, o_supp_max, o_late_min, o_late_max,
    num_lineitems,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_lineitems

    o_key = tl.load(l_orderkey + offs, mask=mask, other=0)
    s_key = tl.load(l_suppkey + offs, mask=mask, other=0)
    receipt = tl.load(l_receiptdate + offs, mask=mask, other=0)
    commit = tl.load(l_commitdate + offs, mask=mask, other=0)

    # Only process lineitems for orders with status 'F'
    is_F = tl.load(order_status_F + o_key, mask=mask, other=0)
    active_mask = mask & (is_F == 1)

    late = receipt > commit

    # All active suppliers per order: atomic min/max
    # Idempotent operation, safe with triton.autotune
    tl.atomic_min(o_supp_min + o_key, s_key, mask=active_mask)
    tl.atomic_max(o_supp_max + o_key, s_key, mask=active_mask)

    # Late active suppliers per order: atomic min/max
    late_mask = active_mask & late
    tl.atomic_min(o_late_min + o_key, s_key, mask=late_mask)
    tl.atomic_max(o_late_max + o_key, s_key, mask=late_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["num_lineitems"],
    reset_to_zero=["counts"],
)
@triton.jit
def filter_and_count(
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
'''


def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q21_triton_")
    path = os.path.join(tmpdir, "_q21_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q21_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _safe_max(sr: cudf.Series) -> int:
    if len(sr) == 0:
        return 0
    return int(sr.max())

def run_query(tables: dict[str, cudf.DataFrame], nation_name: str = "SAUDI ARABIA", orderstatus: str = "F") -> cudf.DataFrame:
    """
    SQL: TPC-H Q21 - Suppliers Who Kept Orders Waiting.
    Description:
      Uses two Triton fused kernels. 
      The first kernel builds per-order statistics using idempotent `atomic_min` and `atomic_max`
      to efficiently track the presence of multiple unique suppliers and multiple unique late suppliers.
      The second kernel evaluates the EXISTS and NOT EXISTS subquery conditions directly 
      using the precomputed min/max states and aggregates the target suppliers count into HBM,
      fully bypassing expensive GroupBy/Merge cuDF operations.
    Params:
      nation_name (str): nation name filter.
      orderstatus (str): order status filter.
    """
    df_supp = tables["supplier"]
    df_li = tables["lineitem"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # --- 1. Filter nation -> supplier (cuDF) ---
    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supp.merge(nation_f, left_on="s_nationkey", right_on="n_nationkey")
    supp_n = supp_n[["s_suppkey", "s_name"]].sort_values("s_name").reset_index(drop=True)

    if len(supp_n) == 0:
        return cudf.DataFrame({"s_name": cudf.Series(dtype="object"), "numwait": cudf.Series(dtype="int32")})

    # Ensure robust max dimensions for static array allocation
    max_orderkey = max(_safe_max(df_li["l_orderkey"]), _safe_max(df_orders["o_orderkey"]))
    max_suppkey = max(_safe_max(df_li["l_suppkey"]), _safe_max(df_supp["s_suppkey"]))

    # --- 2. Build LUTs on GPU ---
    # Map from suppkey to linear index for the matching target suppliers, -1 elsewhere
    target_supp_map = torch.full((max_suppkey + 1,), -1, dtype=torch.int32, device="cuda")
    target_supp_map[torch.as_tensor(supp_n["s_suppkey"].to_cupy(), dtype=torch.int64)] = \
        torch.arange(len(supp_n), dtype=torch.int32, device="cuda")

    # Boolean array of active orders by status
    order_status_F = torch.zeros(max_orderkey + 1, dtype=torch.uint8, device="cuda")
    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus]
    if len(orders_f) > 0:
        order_status_F[torch.as_tensor(orders_f["o_orderkey"].to_cupy(), dtype=torch.int64)] = 1

    # Order statistics states mapped in HBM
    o_supp_min = torch.full((max_orderkey + 1,), 2147483647, dtype=torch.int32, device="cuda")
    o_supp_max = torch.full((max_orderkey + 1,), -1, dtype=torch.int32, device="cuda")
    o_late_min = torch.full((max_orderkey + 1,), 2147483647, dtype=torch.int32, device="cuda")
    o_late_max = torch.full((max_orderkey + 1,), -1, dtype=torch.int32, device="cuda")

    # Extract tensors
    l_orderkey = torch.as_tensor(df_li["l_orderkey"].to_cupy(), dtype=torch.int32)
    l_suppkey = torch.as_tensor(df_li["l_suppkey"].to_cupy(), dtype=torch.int32)
    l_receiptdate = torch.as_tensor((df_li["l_receiptdate"].astype("int64") // 86400).astype("int32").to_cupy())
    l_commitdate = torch.as_tensor((df_li["l_commitdate"].astype("int64") // 86400).astype("int32").to_cupy())

    num_lineitems = l_orderkey.numel()

    # --- 3. Run Fused Triton Kernels ---
    grid = lambda meta: (triton.cdiv(num_lineitems, meta["BLOCK_SIZE"]),)

    _K.build_order_stats[grid](
        l_orderkey, l_suppkey, l_receiptdate, l_commitdate,
        order_status_F,
        o_supp_min, o_supp_max, o_late_min, o_late_max,
        num_lineitems
    )

    counts = torch.zeros(len(supp_n), dtype=torch.int32, device="cuda")

    _K.filter_and_count[grid](
        l_orderkey, l_suppkey, l_receiptdate, l_commitdate,
        o_supp_min, o_supp_max, o_late_min, o_late_max,
        target_supp_map, order_status_F, counts,
        num_lineitems
    )

    # --- 4. Finalize and Return ---
    result = cudf.DataFrame({
        "s_name": supp_n["s_name"],
        "numwait": cudf.Series(counts)
    })
    result = result[result["numwait"] > 0]
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100).reset_index(drop=True)
    return result
