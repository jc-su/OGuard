import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _compute_amount_kernel(
    price_ptr, discount_ptr, cost_ptr, qty_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    cost = tl.load(cost_ptr + offs, mask=mask, other=0.0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    
    amount = price * (1.0 - disc) - cost * qty
    tl.store(out_ptr + offs, amount, mask=mask)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q9_triton_")
    path = os.path.join(tmpdir, "_q9_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q9_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Compute amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity.
    Uses fused Triton kernel for the computation.
    """
    n = l_extendedprice.numel()
    out = torch.empty(n, dtype=torch.float64, device=l_extendedprice.device)
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _K._compute_amount_kernel[grid](
        l_extendedprice, l_discount, ps_supplycost, l_quantity, out, n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def run_query(tables: dict[str, cudf.DataFrame], name_pattern: str = "green") -> cudf.DataFrame:
    """
    SQL: SELECT nation, o_year, sum(amount) AS sum_profit FROM (SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM part, supplier, lineitem, partsupp, orders, nation WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey AND p_name LIKE '%green%') AS profit GROUP BY nation, o_year ORDER BY nation, o_year DESC;
    
    Description: TPC-H Q9 - Product Type Profit Measure. Computes profit by nation and year for parts whose name matches a pattern.
    Uses cuDF for multi-table joins (memory-bound, efficient cuDF implementation) and Triton for the fused arithmetic computation (memory-bound row scan, benefits from single kernel launch).
    
    Params: name_pattern (str): substring to filter part names (without % wildcards).
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    pat = name_pattern.replace("%", "")
    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]

    # cuDF joins
    li_parts = df_lineitem.merge(parts_filtered, left_on="l_partkey", right_on="p_partkey")

    li_ps = li_parts.merge(
        df_partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]],
        left_on=["l_partkey", "l_suppkey"],
        right_on=["ps_partkey", "ps_suppkey"],
    )

    li_ps_s = li_ps.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )

    li_ps_s_o = li_ps_s.merge(
        df_orders[["o_orderkey", "o_orderdate"]],
        left_on="l_orderkey",
        right_on="o_orderkey",
    )

    joined = li_ps_s_o.merge(
        df_nation[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    # Extract needed columns as tensors
    l_extendedprice = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    ps_supplycost = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    l_quantity = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)

    # Compute amount using fused Triton kernel
    amount_t = _query_core(l_extendedprice, l_discount, ps_supplycost, l_quantity)

    # Build profit subquery frame in cuDF
    profit_df = cudf.DataFrame({
        "nation": joined["n_name"].reset_index(drop=True),
        "o_year": joined["o_orderdate"].dt.year.astype("int32").reset_index(drop=True),
        "amount": cudf.Series(amount_t),
    })

    # Group by nation, o_year and sum
    grouped = profit_df.groupby(["nation", "o_year"], as_index=False).agg({"amount": "sum"})
    grouped = grouped.rename(columns={"amount": "sum_profit"})

    # Order by nation ASC, o_year DESC
    grouped = grouped.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)

    return cudf.DataFrame({
        "nation": grouped["nation"],
        "o_year": grouped["o_year"],
        "sum_profit": grouped["sum_profit"],
    })
