import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
import numpy as np

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
def _q5_kernel(
    nation_id_ptr, price_ptr, discount_ptr,
    out_ptr, n_elements, num_nations,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    nation_id = tl.load(nation_id_ptr + offs, mask=mask, other=num_nations)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    revenue = price * (1.0 - discount)
    valid = mask & (nation_id < num_nations)
    
    # Per-nation accumulator using static range for small fixed groups
    # num_nations is a constexpr via passed value (will be compiled per query)
    # But we need it as a compile-time constant for tl.zeros shape
    # Instead, use a loop: accumulate revenue per nation
    for i in tl.static_range(0, 5):  # Will be compiled with actual num_nations
        # Use tl.where and tl.sum per nation
        nation_mask = valid & (nation_id == i)
        contrib = tl.where(nation_mask, revenue, 0.0)
        block_sum = tl.sum(contrib)
        tl.atomic_add(out_ptr + i, block_sum)
    # Handle up to 25 nations max (TPC-H has 25)
    for i in tl.static_range(5, 25):
        nation_mask = valid & (nation_id == i)
        contrib = tl.where(nation_mask, revenue, 0.0)
        block_sum = tl.sum(contrib)
        tl.atomic_add(out_ptr + i, block_sum)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q5_triton_")
    path = os.path.join(tmpdir, "_q5_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q5_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _launch_q5(
    nation_ids: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch fused Triton kernel to aggregate revenue per nation.
    
    Fuses projection (price * (1 - discount)) and scatter-add aggregation
    into a single memory-bound pass, reducing HBM traffic.
    
    Args:
        nation_ids: int64 tensor of nation IDs
        l_extendedprice: float64 price tensor
        l_discount: float64 discount tensor
        num_nations: number of distinct nations (fixed, small, ≤ 25)
    
    Returns:
        sorted_nation_ids, sorted_revenues
    """
    n = nation_ids.numel()
    out = torch.zeros(num_nations, dtype=torch.float64, device=nation_ids.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q5_kernel[grid](
        nation_ids, l_extendedprice, l_discount,
        out, n, num_nations,
    )
    # Sort by revenue descending
    order = torch.argsort(out, descending=True)
    nation_range = torch.arange(num_nations, dtype=torch.int64, device=nation_ids.device)
    return nation_range[order], out[order]


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
    Optimized path: cuDF handles all join operations and string work; the hot
    path (project revenue + aggregate per nation) is fused into a single Triton
    kernel that reads nation_id, price, and discount from HBM exactly once.
    This eliminates the intermediate 'revenue' tensor and scatter_add call.
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

    # Filter region (tiny table)
    region_f = df_region[df_region["r_name"] == region_name]
    # Nation within region (tiny, ≤ 25 rows)
    nation_f = df_nation.merge(region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey")

    # Orders filtered by date range (filter before join to reduce data)
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (orders_f["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = orders_f[
        (orders_f["o_orderdate_days"] >= lo_days)
        & (orders_f["o_orderdate_days"] < hi_days)
    ]

    # Join customer with orders on custkey
    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f[["o_orderkey", "o_custkey"]], left_on="c_custkey", right_on="o_custkey"
    )

    # Join with lineitem on orderkey
    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey", right_on="l_orderkey",
    )

    # Join with supplier on suppkey AND enforce c_nationkey = s_nationkey
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey", right_on="s_suppkey",
    )
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    # Join with filtered nation on s_nationkey = n_nationkey
    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey", right_on="n_nationkey",
    )

    # Build nation_id mapping
    nations_unique = nation_f[["n_nationkey", "n_name"]].drop_duplicates().sort_values("n_nationkey").reset_index(drop=True)
    nations_unique["nation_id"] = cudf.Series(range(len(nations_unique)), dtype="int64")

    final_df = final_df.merge(nations_unique[["n_nationkey", "nation_id"]], on="n_nationkey")

    num_nations = int(len(nations_unique))

    # Convert to GPU tensors
    nation_ids_t = torch.as_tensor(final_df["nation_id"].to_cupy(), dtype=torch.int64)
    price_t = torch.as_tensor(final_df["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(final_df["l_discount"].to_cupy(), dtype=torch.float64)

    # Fused kernel launch: project + aggregate in one pass
    nation_ids_out, revenue_out = _launch_q5(
        nation_ids_t, price_t, discount_t, num_nations,
    )

    # Map back nation_id to n_name
    result_ids = cudf.Series(nation_ids_out, dtype="int64")
    result_rev = cudf.Series(revenue_out, dtype="float64")
    result = cudf.DataFrame({"nation_id": result_ids, "revenue": result_rev})
    result = result.merge(nations_unique[["nation_id", "n_name"]], on="nation_id")
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame({
        "n_name": result["n_name"],
        "revenue": result["revenue"],
    })
