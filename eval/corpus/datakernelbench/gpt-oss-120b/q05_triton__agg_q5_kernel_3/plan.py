"""
Optimized Q5 implementation using Triton fusion.

SQL:
SELECT
    n_name,
    sum(l_extendedprice * (1 - l_discount)) AS revenue
FROM
    customer, orders, lineitem, supplier, nation, region
WHERE
    c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND l_suppkey = s_suppkey
    AND c_nationkey = s_nationkey
    AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey
    AND r_name = 'ASIA'
    AND o_orderdate >= DATE '1994-01-01'
    AND o_orderdate < DATE '1995-01-01'
GROUP BY n_name
ORDER BY revenue DESC;

Description:
- All heavy joins and string handling stay in cuDF (which is already highly
  optimized for GPU).
- The aggregation (revenue per nation) is fused into a single Triton kernel.
  The kernel streams nation_id, price and discount, computes the revenue and
  atomically adds it to a per‑nation accumulator.  This eliminates the
  scatter_add overhead and reduces memory traffic to a single pass.
- After the kernel we sort the per‑nation revenues with a single torch
  argsort, keeping everything on the GPU.

Params:
  region_name (str): region filter (default 'ASIA')
  orderdate_lo (str): inclusive lower bound (YYYY‑MM‑DD)
  orderdate_hi (str): exclusive upper bound (YYYY‑MM‑DD)
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
# Triton kernel source – must be written to disk before import
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _agg_q5_kernel(
    nation_ptr,          # int64 *
    price_ptr,           # float64 *
    discount_ptr,        # float64 *
    out_ptr,             # float64 * (size = num_nations)
    n_elements,          # int64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load inputs; masked lanes get benign values
    nid = tl.load(nation_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Compute revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)

    # Zero out contributions from masked lanes to avoid spurious atomics
    revenue = tl.where(mask, revenue, 0.0)
    nid = tl.where(mask, nid, 0)

    # Atomic add per nation
    tl.atomic_add(out_ptr + nid, revenue)
'''

def _load_kernels():
    """Write the kernel source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="q5_triton_")
    path = os.path.join(tmpdir, "_q5_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q5_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load once at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Helper to launch the fused aggregation kernel
# ----------------------------------------------------------------------
def _launch_agg_q5(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> torch.Tensor:
    """Fuse filter‑free projection + revenue aggregation via Triton."""
    n = nation_id.numel()
    out = torch.zeros(num_nations, dtype=torch.float64, device=nation_id.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._agg_q5_kernel[grid](
        nation_id,
        l_extendedprice,
        l_discount,
        out,
        n,
    )
    return out

# ----------------------------------------------------------------------
# Public core API (kept for compatibility)
# ----------------------------------------------------------------------
def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per‑nation revenue using a fused Triton kernel,
    then return sorted (nation_id, revenue) in descending revenue order.
    """
    revenue = _launch_agg_q5(nation_id, l_extendedprice, l_discount, num_nations)

    nation_ids = torch.arange(num_nations, dtype=torch.int64, device=revenue.device)
    order = torch.argsort(revenue, descending=True)
    return nation_ids[order], revenue[order]

# ----------------------------------------------------------------------
# End‑user entry point (unchanged signature)
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "ASIA",
    orderdate_lo: str = "1994-01-01",
    orderdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    Execute TPC‑H Q5 on the provided cuDF tables.

    Parameters
    ----------
    tables : dict[str, cudf.DataFrame]
        Mapping from table name to loaded cuDF DataFrame.
    region_name : str, optional
        Region to filter on (default 'ASIA').
    orderdate_lo : str, optional
        Inclusive lower bound for o_orderdate (YYYY‑MM‑DD).
    orderdate_hi : str, optional
        Exclusive upper bound for o_orderdate (YYYY‑MM‑DD).

    Returns
    -------
    cudf.DataFrame
        Columns ``n_name`` and ``revenue`` sorted by revenue descending.
    """
    # ------------------------------------------------------------------
    # Load tables
    # ------------------------------------------------------------------
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    # ------------------------------------------------------------------
    # Date boundaries as epoch days (int)
    # ------------------------------------------------------------------
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # ------------------------------------------------------------------
    # Region → Nation filter
    # ------------------------------------------------------------------
    region_f = df_region[df_region["r_name"] == region_name]
    nation_f = df_nation.merge(
        region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey"
    )

    # ------------------------------------------------------------------
    # Orders filtered by orderdate
    # ------------------------------------------------------------------
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (
        (orders_f["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    orders_f = orders_f[
        (orders_f["o_orderdate_days"] >= lo_days)
        & (orders_f["o_orderdate_days"] < hi_days)
    ]

    # ------------------------------------------------------------------
    # Join pipeline (still in cuDF)
    # ------------------------------------------------------------------
    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f[["o_orderkey", "o_custkey"]], left_on="c_custkey", right_on="o_custkey"
    )
    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey",
        right_on="l_orderkey",
    )
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )
    # Enforce c_nationkey == s_nationkey
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    # ------------------------------------------------------------------
    # Build deterministic nation_id mapping (dense 0..num_nations-1)
    # ------------------------------------------------------------------
    nations_unique = (
        nation_f[["n_nationkey", "n_name"]]
        .drop_duplicates()
        .sort_values("n_nationkey")
        .reset_index(drop=True)
    )
    nations_unique["nation_id"] = cudf.Series(
        range(len(nations_unique)), dtype="int64"
    )
    final_df = final_df.merge(
        nations_unique[["n_nationkey", "nation_id"]], on="n_nationkey"
    )
    num_nations = int(len(nations_unique))

    # ------------------------------------------------------------------
    # Prepare tensors for the fused aggregation
    # ------------------------------------------------------------------
    nation_id_t = torch.as_tensor(
        final_df["nation_id"].to_cupy(), dtype=torch.int64
    )
    price_t = torch.as_tensor(
        final_df["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    discount_t = torch.as_tensor(
        final_df["l_discount"].to_cupy(), dtype=torch.float64
    )

    # ------------------------------------------------------------------
    # Compute per‑nation revenue (fused kernel) and order results
    # ------------------------------------------------------------------
    nation_ids_t, revenue_t = _query_core(
        nation_id_t, price_t, discount_t, num_nations
    )

    # Map back to nation names
    result_ids = cudf.Series(nation_ids_t, dtype="int64")
    result_rev = cudf.Series(revenue_t, dtype="float64")
    result = cudf.DataFrame(
        {"nation_id": result_ids, "revenue": result_rev}
    )
    result = result.merge(
        nations_unique[["nation_id", "n_name"]], on="nation_id"
    )
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame({"n_name": result["n_name"], "revenue": result["revenue"]})
