import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ============================================================================
# Triton kernel source: elementwise amount = price * (1 - discount) - supplycost * quantity
# ============================================================================
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
)
@triton.jit
def _amount_kernel(
    price_ptr,
    discount_ptr,
    supplycost_ptr,
    quantity_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute amount = price * (1 - discount) - supplycost * quantity per row."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input columns (float64)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    # Elementwise profit formula
    amount = price * (1.0 - discount) - supplycost * quantity

    # Store result
    tl.store(out_ptr + offs, amount, mask=mask)
'''

# -------------------------------------------------------------------------
# Helper to load Triton kernels from the source string
# -------------------------------------------------------------------------
def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q9_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

# -------------------------------------------------------------------------
# Launch the fused elementwise kernel
# -------------------------------------------------------------------------
def _launch_amount(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """Launch Triton kernel to compute amount per row."""
    n = l_extendedprice.numel()
    # Allocate output tensor (float64) with same shape
    out = torch.empty_like(l_extendedprice)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._amount_kernel[grid](
        l_extendedprice,
        l_discount,
        ps_supplycost,
        l_quantity,
        out,
        n,
    )
    return out

# -------------------------------------------------------------------------
# Core compute: elementwise amount (matches reference signature)
# -------------------------------------------------------------------------
def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Compute amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity.
    Returns a 1‑D float64 tensor with the per‑row profit values.
    """
    return _launch_amount(l_extendedprice, l_discount, ps_supplycost, l_quantity)

# -------------------------------------------------------------------------
# Query execution: join, compute, aggregate (identical to reference)
# -------------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    name_pattern: str = "green",
) -> cudf.DataFrame:
    """
    SQL: SELECT nation, o_year, sum(amount) AS sum_profit FROM (...) GROUP BY nation, o_year ORDER BY nation, o_year DESC;
    Description: TPC‑H Q9 – Product Type Profit Measure. Computes profit by nation and year for parts whose name matches a pattern.
    Params:
        tables: dict of cuDF DataFrames (part, supplier, lineitem, partsupp, orders, nation).
        name_pattern: substring to filter part names (without % wildcards).
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # Filter parts by name pattern
    pat = name_pattern.replace("%", "")
    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]

    # Join chain: lineitem → parts → partsupp → supplier → orders → nation
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

    # Compute amount using the optimized Triton kernel
    amount_t = _query_core(
        torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64),
    )

    # Build profit DataFrame and aggregate
    profit_df = cudf.DataFrame({
        "nation": joined["n_name"].reset_index(drop=True),
        "o_year": joined["o_orderdate"].dt.year.astype("int32").reset_index(drop=True),
        "amount": cudf.Series(amount_t),
    })

    grouped = profit_df.groupby(["nation", "o_year"], as_index=False).agg({"amount": "sum"})
    grouped = grouped.rename(columns={"amount": "sum_profit"})
    grouped = grouped.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)

    return cudf.DataFrame({
        "nation": grouped["nation"],
        "o_year": grouped["o_year"],
        "sum_profit": grouped["sum_profit"],
    })
