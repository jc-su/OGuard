import os
import tempfile
import importlib.util
import linecache
from typing import Any

import cudf
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q21_count_kernel(
    cand_orderkey_ptr,
    cand_supp_id_ptr,
    cand_late_count_ptr,
    stat_orderkey_ptr,
    stat_distinct_ptr,
    stat_late_distinct_ptr,
    counts_ptr,
    n_cand,
    n_stats,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cand

    key = tl.load(cand_orderkey_ptr + offs, mask=mask, other=0)
    supp_id = tl.load(cand_supp_id_ptr + offs, mask=mask, other=0)
    late_count = tl.load(cand_late_count_ptr + offs, mask=mask, other=0)

    # Vectorized lower_bound over sorted per-order statistics.
    lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    hi = lo + n_stats
    hi = tl.where(mask, hi, lo)

    for _ in tl.static_range(0, 32):
        active = lo < hi
        mid = (lo + hi) // 2
        mid_key = tl.load(stat_orderkey_ptr + mid, mask=active, other=9223372036854775807)
        go_right = mid_key < key
        lo = tl.where(active & go_right, mid + 1, lo)
        hi = tl.where(active & ~go_right, mid, hi)

    in_range = mask & (lo < n_stats)
    found_key = tl.load(stat_orderkey_ptr + lo, mask=in_range, other=-1)
    found = in_range & (found_key == key)

    distinct_supp = tl.load(stat_distinct_ptr + lo, mask=found, other=0)
    late_distinct_supp = tl.load(stat_late_distinct_ptr + lo, mask=found, other=2)

    keep = found & (late_count > 0) & (distinct_supp >= 2) & (late_distinct_supp <= 1)
    tl.atomic_add(counts_ptr + supp_id, late_count, sem="relaxed", mask=keep)
'''


def _load_kernels() -> Any:
    """Write Triton JIT kernels to a real file and import them for validator-safe compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q21_triton_")
    path = os.path.join(tmpdir, "_q21_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q21_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _empty_result(name_template: cudf.Series) -> cudf.DataFrame:
    """Build an empty cuDF result with the SQL output schema."""
    return cudf.DataFrame(
        {
            "s_name": name_template.head(0).reset_index(drop=True),
            "numwait": cudf.Series([], dtype="int64"),
        }
    )


def _to_i64_tensor(series: cudf.Series) -> torch.Tensor:
    """Convert a cuDF numeric column to a GPU torch int64 tensor without a CPU data copy."""
    return torch.as_tensor(series.to_cupy(), dtype=torch.int64)


def _launch_q21_count(
    cand_orderkey: torch.Tensor,
    cand_supp_id: torch.Tensor,
    cand_late_count: torch.Tensor,
    stat_orderkey: torch.Tensor,
    stat_distinct_supp: torch.Tensor,
    stat_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Run the fused GPU count kernel.

    The Triton kernel performs, in one pass over candidate (order, supplier) rows, the sorted
    order-stat lookup, Q21 EXISTS / NOT EXISTS predicates, and atomic grouped count by supplier.
    This avoids materializing a candidate/order-stats join and separate filter/scatter tensors.
    """
    device = cand_orderkey.device if cand_orderkey.numel() else stat_orderkey.device
    counts = torch.zeros(num_suppliers, dtype=torch.int64, device=device)

    n_cand = int(cand_orderkey.numel())
    n_stats = int(stat_orderkey.numel())
    if n_cand == 0 or n_stats == 0 or num_suppliers == 0:
        return counts

    grid = lambda meta: (triton.cdiv(n_cand, meta["BLOCK_SIZE"]),)
    _K._q21_count_kernel[grid](
        cand_orderkey,
        cand_supp_id,
        cand_late_count,
        stat_orderkey,
        stat_distinct_supp,
        stat_late_distinct_supp,
        counts,
        n_cand,
        n_stats,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return counts


def _query_core(
    cand_orderkey: torch.Tensor,
    cand_supp_id: torch.Tensor,
    cand_late_count: torch.Tensor,
    stat_orderkey: torch.Tensor,
    stat_distinct_supp: torch.Tensor,
    stat_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU core for Q21 final aggregation.

    Candidate rows are already prepared by cuDF joins/groupbys. Triton then fuses order-stat
    lookup, predicates, and supplier-count aggregation into one memory-bound GPU pass.
    """
    counts = _launch_q21_count(
        cand_orderkey,
        cand_supp_id,
        cand_late_count,
        stat_orderkey,
        stat_distinct_supp,
        stat_late_distinct_supp,
        num_suppliers,
    )
    return torch.arange(num_suppliers, dtype=torch.int64, device=counts.device), counts


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "SAUDI ARABIA",
    orderstatus: str = "F",
) -> cudf.DataFrame:
    """
    SQL:
      SELECT s_name, count(*) AS numwait
      FROM supplier, lineitem l1, orders, nation
      WHERE s_suppkey = l1.l_suppkey
        AND o_orderkey = l1.l_orderkey
        AND o_orderstatus = 'F'
        AND l1.l_receiptdate > l1.l_commitdate
        AND EXISTS (
          SELECT * FROM lineitem l2
          WHERE l2.l_orderkey = l1.l_orderkey
            AND l2.l_suppkey <> l1.l_suppkey)
        AND NOT EXISTS (
          SELECT * FROM lineitem l3
          WHERE l3.l_orderkey = l1.l_orderkey
            AND l3.l_suppkey <> l1.l_suppkey
            AND l3.l_receiptdate > l3.l_commitdate)
        AND s_nationkey = n_nationkey
        AND n_name = 'SAUDI ARABIA'
      GROUP BY s_name
      ORDER BY numwait DESC, s_name
      LIMIT 100;

    Description:
      cuDF handles string filters, joins, and variable-size groupbys: it filters orders by status,
      filters suppliers by nation, and collapses the order/supplier lineitems into per-pair
      late-line counts. It then derives per-order distinct-supplier and late-distinct-supplier
      statistics. The hot final candidate evaluation runs in Triton: one fused GPU kernel scans
      late candidate supplier/order pairs, binary-searches sorted order stats, applies the
      EXISTS / NOT EXISTS predicates in registers, and atomically accumulates int64 counts by
      supplier. This reduces intermediate materialization versus a separate join/filter/scatter
      tensor pipeline and keeps the final memory-bound pass coalesced for candidate columns.

    Params:
      tables (dict[str, cudf.DataFrame]): TPC-H tables keyed by table name.
      nation_name (str): Nation name filter, default "SAUDI ARABIA".
      orderstatus (str): Order status filter, default "F".
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    nation_keys = df_nation[df_nation["n_name"] == nation_name]["n_nationkey"]
    if len(nation_keys) == 0:
        return _empty_result(df_supplier["s_name"])

    nation_key = int(nation_keys.iloc[0])
    supp_n = df_supplier[df_supplier["s_nationkey"] == nation_key][["s_suppkey", "s_name"]]
    supp_n = supp_n.sort_values("s_name").reset_index(drop=True)
    num_suppliers = len(supp_n)
    if num_suppliers == 0:
        return _empty_result(df_supplier["s_name"])

    supp_n["supp_id"] = cudf.Series(range(num_suppliers), dtype="int32")

    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]
    if len(orders_f) == 0:
        return _empty_result(supp_n["s_name"])

    li_cols = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]]
    li_f = li_cols.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey", how="inner")
    if len(li_f) == 0:
        return _empty_result(supp_n["s_name"])

    li_f["late_i"] = (li_f["l_receiptdate"] > li_f["l_commitdate"]).astype("int32")
    pair_src = li_f[["l_orderkey", "l_suppkey", "late_i"]]
    del li_f

    pair = pair_src.groupby(["l_orderkey", "l_suppkey"]).agg({"late_i": "sum"}).reset_index()
    pair = pair.rename(columns={"late_i": "late_count"})
    if len(pair) == 0:
        return _empty_result(supp_n["s_name"])

    pair["has_late"] = (pair["late_count"] > 0).astype("int32")
    stats = pair.groupby("l_orderkey").agg({"l_suppkey": "count", "has_late": "sum"}).reset_index()
    stats = stats.rename(
        columns={
            "l_suppkey": "distinct_supp",
            "has_late": "late_distinct_supp",
        }
    )
    stats = stats.sort_values("l_orderkey").reset_index(drop=True)

    pair_late = pair[pair["late_count"] > 0][["l_orderkey", "l_suppkey", "late_count"]]
    if len(pair_late) == 0:
        return _empty_result(supp_n["s_name"])

    supp_map = supp_n[["s_suppkey", "supp_id"]]
    candidates = pair_late.merge(
        supp_map,
        left_on="l_suppkey",
        right_on="s_suppkey",
        how="inner",
    )
    if len(candidates) == 0:
        return _empty_result(supp_n["s_name"])

    cand_orderkey_t = _to_i64_tensor(candidates["l_orderkey"])
    cand_supp_id_t = _to_i64_tensor(candidates["supp_id"])
    cand_late_count_t = _to_i64_tensor(candidates["late_count"])

    stat_orderkey_t = _to_i64_tensor(stats["l_orderkey"])
    stat_distinct_t = _to_i64_tensor(stats["distinct_supp"])
    stat_late_distinct_t = _to_i64_tensor(stats["late_distinct_supp"])

    _, counts_t = _query_core(
        cand_orderkey_t,
        cand_supp_id_t,
        cand_late_count_t,
        stat_orderkey_t,
        stat_distinct_t,
        stat_late_distinct_t,
        num_suppliers,
    )

    result = cudf.DataFrame()
    result["s_name"] = supp_n["s_name"].reset_index(drop=True)
    result["numwait"] = cudf.Series(counts_t)
    result = result[result["numwait"] > 0]
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100)
    return result.reset_index(drop=True)
