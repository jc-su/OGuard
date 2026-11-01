import os
import tempfile
import importlib.util
import linecache

import torch
import triton
import cudf
import pandas as pd

# --------------------------------------------------------------------------
# Triton kernel source: fused per‑nation revenue aggregation
# --------------------------------------------------------------------------
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
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _agg_revenue_kernel(
    nation_id_ptr,
    price_ptr,
    discount_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: compute revenue = price * (1 - discount) and atomic‑add into per‑nation buckets."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load row data (int64 nation_id, float64 price, float64 discount)
    nation_id = tl.load(nation_id_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Compute revenue for valid rows
    revenue = price * (1.0 - discount)

    # Atomic add to the per‑nation accumulator; zero contribution for masked rows
    tl.atomic_add(out_ptr + nation_id, tl.where(mask, revenue, 0.0))
'''

def _load_kernels():
    """Write kernel source to a temp file and load it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="q5_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q5_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def _launch_agg_revenue(
    nation_ids: torch.Tensor,
    prices: torch.Tensor,
    discounts: torch.Tensor,
    num_nations: int,
) -> torch.Tensor:
    """Launch the Triton kernel to compute per‑nation revenue sums.

    This fused kernel reduces memory traffic by performing the revenue
    projection and the group‑by aggregation in a single pass.
    """
    device = prices.device
    n_rows = nation_ids.numel()
    # Allocate and zero‑initialize output (autotune will reset before each trial)
    out = torch.zeros(num_nations, dtype=torch.float64, device=device)

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _K._agg_revenue_kernel[grid](
        nation_ids,
        prices,
        discounts,
        out,
        n_rows,
    )
    return out


def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute revenue per nation and return sorted results.

    Parameters
    ----------
    nation_id : torch.Tensor
        1‑D int64 tensor of nation identifiers (0‑based indices).
    l_extendedprice : torch.Tensor
        1‑D float64 tensor of line item extended prices.
    l_discount : torch.Tensor
        1‑D float64 tensor of line item discounts.
    num_nations : int
        Number of distinct nations (size of output vector).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        First tensor: nation identifiers sorted by descending revenue.
        Second tensor: corresponding revenue totals.
    """
    # Aggregate revenue using Triton kernel
    revenue_per_nation = _launch_agg_revenue(
        nation_id, l_extendedprice, l_discount, num_nations
    )
    # Sort by revenue descending
    order = torch.argsort(revenue_per_nation, descending=True)
    nation_ids_ordered = torch.arange(
        num_nations, dtype=torch.int64, device=revenue_per_nation.device
    )
    return nation_ids_ordered[order], revenue_per_nation[order]


def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "ASIA",
    orderdate_lo: str = "1994-01-01",
    orderdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
         FROM customer, orders, lineitem, supplier, nation, region
         WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
           AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
           AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
           AND r_name = 'ASIA'
           AND o_orderdate >= DATE '1994-01-01'
           AND o_orderdate <  DATE '1995-01-01'
         GROUP BY n_name ORDER BY revenue DESC;
    Description: TPC-H Q5 — Local supplier volume for a region over one year.
    Params:
      region_name (str): region to filter on.
      orderdate_lo (str): inclusive lower bound on order date.
      orderdate_hi (str): exclusive upper bound on order date.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter region
    region_f = df_region[df_region["r_name"] == region_name]
    # Nation within region
    nation_f = df_nation.merge(
        region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey"
    )

    # Orders filtered by date
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (
        orders_f["o_orderdate"].astype("int64") // 86400
    ).astype("int32")
    orders_f = orders_f[
        (orders_f["o_orderdate_days"] >= lo_days)
        & (orders_f["o_orderdate_days"] < hi_days)
    ]

    # Join customer with orders on custkey
    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f[["o_orderkey", "o_custkey"]],
        left_on="c_custkey",
        right_on="o_custkey",
    )

    # Join with lineitem on orderkey
    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey",
        right_on="l_orderkey",
    )

    # Join with supplier on suppkey AND c_nationkey = s_nationkey
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    # Join with filtered nation on s_nationkey = n_nationkey
    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    # Build nation_id mapping from distinct nations in filter
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

    nation_ids_t, revenue_t = _query_core(
        torch.as_tensor(final_df["nation_id"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(final_df["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(final_df["l_discount"].to_cupy(), dtype=torch.float64),
        num_nations,
    )

    # Map back nation_id to n_name
    result_ids = cudf.Series(nation_ids_t, dtype="int64")
    result_rev = cudf.Series(revenue_t, dtype="float64")
    result = cudf.DataFrame({"nation_id": result_ids, "revenue": result_rev})
    result = result.merge(
        nations_unique[["nation_id", "n_name"]], on="nation_id"
    )
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame({"n_name": result["n_name"], "revenue": result["revenue"]})
