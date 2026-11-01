"""
TPC‑H Q21 – Suppliers Who Kept Orders Waiting – Triton‑fused implementation.

SQL:
SELECT
    s_name,
    count(*) AS numwait
FROM
    supplier,
    lineitem l1,
    orders,
    nation
WHERE
    s_suppkey = l1.l_suppkey
    AND o_orderkey = l1.l_orderkey
    AND o_orderstatus = 'F'
    AND l1.l_receiptdate > l1.l_commitdate
    AND EXISTS (
        SELECT *
        FROM lineitem l2
        WHERE l2.l_orderkey = l1.l_orderkey
          AND l2.l_suppkey <> l1.l_suppkey)
    AND NOT EXISTS (
        SELECT *
        FROM lineitem l3
        WHERE l3.l_orderkey = l1.l_orderkey
          AND l3.l_suppkey <> l1.l_suppkey
          AND l3.l_receiptdate > l3.l_commitdate)
    AND s_nationkey = n_nationkey
    AND n_name = 'SAUDI ARABIA'
GROUP BY s_name
ORDER BY numwait DESC, s_name
LIMIT 100;
Description:
  • cuDF handles joins and the per‑order aggregates (small tables).
  • The hot path – scanning the filtered l1 rows and counting qualifying
    orders per supplier – is fused into a single Triton kernel that performs
    the predicate logic and an atomic increment to a per‑supplier counter.
    This removes Python‑level loops, reduces memory traffic, and avoids
    multiple kernel launches.
Params:
  nation_name (str): nation filter (default "SAUDI ARABIA").
  orderstatus (str): order status filter (default "F").
"""

import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

# ----------------------------------------------------------------------
# Triton kernel source – written to a temporary file at import time.
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ] + [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=8, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_rows"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q21_kernel(
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
'''

def _load_kernels() -> object:
    """Write Triton source to a temporary .py file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="q21_triton_")
    path = os.path.join(tmpdir, "_q21_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q21_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod

# Load kernels once at import time.
_K = _load_kernels()

# ----------------------------------------------------------------------
# Triton launch wrapper
# ----------------------------------------------------------------------
def _launch_q21(
    supp_id: torch.Tensor,
    late: torch.Tensor,
    distinct: torch.Tensor,
    late_distinct: torch.Tensor,
    num_suppliers: int,
) -> torch.Tensor:
    """
    Execute the fused kernel that atomically increments per‑supplier
    counters and returns a histogram of length `num_suppliers`.
    """
    n_rows = supp_id.numel()
    device = supp_id.device

    out = torch.zeros(num_suppliers, dtype=torch.int64, device=device)

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _K._q21_kernel[grid](
        supp_id,
        late,
        distinct,
        late_distinct,
        out,
        n_rows,
    )
    return out

# ----------------------------------------------------------------------
# Core query logic (matches reference API)
# ----------------------------------------------------------------------
def _query_core(
    supp_id: torch.Tensor,
    late: torch.Tensor,
    distinct: torch.Tensor,
    late_distinct: torch.Tensor,
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (supplier_index_tensor, count_tensor) where `count_tensor`
    contains the per‑supplier `numwait` values.
    """
    counts = _launch_q21(supp_id, late, distinct, late_distinct, num_suppliers)
    indices = torch.arange(num_suppliers, device=counts.device, dtype=torch.int64)
    return indices, counts

# ----------------------------------------------------------------------
# Public entry point – signature unchanged.
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "SAUDI ARABIA",
    orderstatus: str = "F",
) -> cudf.DataFrame:
    """
    Execute TPC‑H Q21 and return the top‑100 suppliers with the highest
    `numwait` count.

    Parameters
    ----------
    tables : dict[str, cudf.DataFrame]
        Mapping of table name to pre‑loaded cuDF DataFrames.
    nation_name : str, default "SAUDI ARABIA"
        Filter on n_name.
    orderstatus : str, default "F"
        Filter on o_orderstatus.

    Returns
    -------
    cudf.DataFrame
        Columns: s_name (string), numwait (int64) – ordered as in the SQL.
    """
    # --------------------------------------------------------------
    # 1. Supplier dimension (filter by nation, assign compact IDs)
    # --------------------------------------------------------------
    df_nation = tables["nation"]
    df_supplier = tables["supplier"]
    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supplier.merge(
        nation_f, left_on="s_nationkey", right_on="n_nationkey"
    )
    supp_n = supp_n[["s_suppkey", "s_name"]].sort_values("s_name").reset_index(drop=True)
    supp_n["supp_id"] = cudf.Series(range(len(supp_n)), dtype="int32")

    # --------------------------------------------------------------
    # 2. Orders filtered by status
    # --------------------------------------------------------------
    df_orders = tables["orders"]
    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]

    # --------------------------------------------------------------
    # 3. Per‑order aggregates over the full lineitem (restricted to orders_f)
    # --------------------------------------------------------------
    df_lineitem = tables["lineitem"]
    li_all = df_lineitem[
        ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]
    ].merge(
        orders_f, left_on="l_orderkey", right_on="o_orderkey"
    )
    li_all["late"] = (li_all["l_receiptdate"] > li_all["l_commitdate"]).astype("int8")

    # distinct suppliers per orderkey
    distinct_supp = (
        li_all[["l_orderkey", "l_suppkey"]]
        .drop_duplicates()
        .groupby("l_orderkey")
        .size()
        .reset_index()
    )
    distinct_supp.columns = ["l_orderkey", "distinct_supp"]

    # distinct late suppliers per orderkey
    late_rows = li_all[li_all["late"] == 1][["l_orderkey", "l_suppkey"]].drop_duplicates()
    late_distinct = (
        late_rows.groupby("l_orderkey")
        .size()
        .reset_index()
        .rename(columns={0: "late_distinct_supp"})
    )
    late_distinct.columns = ["l_orderkey", "late_distinct_supp"]

    agg = distinct_supp.merge(late_distinct, on="l_orderkey", how="left")
    agg["late_distinct_supp"] = agg["late_distinct_supp"].fillna(0).astype("int32")
    agg["distinct_supp"] = agg["distinct_supp"].astype("int32")
    agg = agg.sort_values("l_orderkey").reset_index(drop=True)

    # --------------------------------------------------------------
    # 4. Build l1 rows (filtered lineitems) and attach aggregates
    # --------------------------------------------------------------
    l1 = df_lineitem[
        ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]
    ].merge(
        supp_n, left_on="l_suppkey", right_on="s_suppkey"
    )
    l1 = l1.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")
    l1["late"] = (l1["l_receiptdate"] > l1["l_commitdate"]).astype("bool")
    l1 = l1.merge(
        agg, left_on="l_orderkey", right_on="l_orderkey", how="left"
    )
    l1["distinct_supp"] = l1["distinct_supp"].fillna(0).astype("int32")
    l1["late_distinct_supp"] = l1["late_distinct_supp"].fillna(0).astype("int32")

    # --------------------------------------------------------------
    # 5. Convert relevant columns to GPU tensors
    # --------------------------------------------------------------
    device = torch.device("cuda")
    supp_id_t = torch.as_tensor(l1["supp_id"].to_cupy(), dtype=torch.int32, device=device)
    late_t = torch.as_tensor(l1["late"].to_cupy(), dtype=torch.int8, device=device)
    distinct_t = torch.as_tensor(l1["distinct_supp"].to_cupy(), dtype=torch.int32, device=device)
    late_dist_t = torch.as_tensor(l1["late_distinct_supp"].to_cupy(), dtype=torch.int32, device=device)

    num_suppliers = len(supp_n)

    # --------------------------------------------------------------
    # 6. Run fused kernel and build final DataFrame
    # --------------------------------------------------------------
    supp_idx_t, counts_t = _query_core(
        supp_id_t,
        late_t,
        distinct_t,
        late_dist_t,
        num_suppliers,
    )

    result = cudf.DataFrame(
        {
            "s_name": supp_n["s_name"]
            .iloc[cudf.Series(supp_idx_t)]
            .reset_index(drop=True),
            "numwait": cudf.Series(counts_t),
        }
    )
    result = result[result["numwait"] > 0]
    result = result.sort_values(
        ["numwait", "s_name"], ascending=[False, True]
    ).head(100).reset_index(drop=True)
    return result
