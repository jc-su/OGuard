import os
import tempfile
import importlib.util
import linecache
import torch
import triton  # needed for grid calculation
import cudf
import pandas as pd

# -------------------------------------------------
# Triton kernels (must be written to disk before use)
# -------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune configurations tuned for H100
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ] + [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=8, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
)
@triton.jit
def _amount_kernel(
    ext_ptr,          # float64 *
    disc_ptr,         # float64 *
    sup_ptr,          # float64 *
    qty_ptr,          # float64 *
    out_ptr,          # float64 *
    n_elements,       # i64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load each column; OOB lanes receive a benign value
    ext = tl.load(ext_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    sup = tl.load(sup_ptr + offs, mask=mask, other=0.0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)

    # amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
    amt = ext * (1.0 - disc) - sup * qty

    tl.store(out_ptr + offs, amt, mask=mask)
'''

def _load_kernels() -> object:
    """Write the Triton source to a temporary file and load it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q9_triton_")
    path = os.path.join(tmpdir, "_q9_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q9_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod

# Load kernels at import time (before any run_query call)
_K = _load_kernels()

# -------------------------------------------------
# Launcher for the fused amount computation
# -------------------------------------------------
def _launch_amount(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the per‑row profit amount on the GPU using a single Triton kernel.
    All tensors must be contiguous, on the same device, and of dtype float64.
    Returns a new tensor containing the amounts.
    """
    n = l_extendedprice.numel()
    out = torch.empty(n, dtype=torch.float64, device=l_extendedprice.device)

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

# -------------------------------------------------
# Core query computation (public API unchanged)
# -------------------------------------------------
def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Public core function that mirrors the reference implementation.
    It now delegates the element‑wise profit calculation to a fused Triton kernel.
    """
    return _launch_amount(l_extendedprice, l_discount, ps_supplycost, l_quantity)

# -------------------------------------------------
# End‑to‑end query
# -------------------------------------------------
def run_query(tables: dict[str, cudf.DataFrame], name_pattern: str = "green") -> cudf.DataFrame:
    """
    SQL:
    SELECT nation, o_year, sum(amount) AS sum_profit
    FROM (
        SELECT n_name AS nation,
               extract(year FROM o_orderdate) AS o_year,
               l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
        FROM part, supplier, lineitem, partsupp, orders, nation
        WHERE s_suppkey = l_suppkey
          AND ps_suppkey = l_suppkey
          AND ps_partkey = l_partkey
          AND p_partkey = l_partkey
          AND o_orderkey = l_orderkey
          AND s_nationkey = n_nationkey
          AND p_name LIKE '%green%')
    AS profit
    GROUP BY nation, o_year
    ORDER BY nation, o_year DESC;

    Description:
    The query joins six tables using cuDF (handling string predicates and joins efficiently on GPU).
    The heavy per‑row arithmetic (profit amount) is fused into a single Triton kernel via _query_core,
    eliminating the intermediate PyTorch tensor operations and reducing memory traffic.
    After the amount column is materialized, a cuDF group‑by aggregates the result.

    Params:
    name_pattern (str): substring to filter part names (without % wildcards).
    """
    # -----------------------------------------------------------------
    # 1️⃣ Prepare tables
    # -----------------------------------------------------------------
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # -----------------------------------------------------------------
    # 2️⃣ Apply the string filter on part name (cuDF string operation)
    # -----------------------------------------------------------------
    pat = name_pattern.replace("%", "")
    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]

    # -----------------------------------------------------------------
    # 3️⃣ Perform the series of joins (all executed on the GPU)
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # 4️⃣ Compute the amount column with the fused Triton kernel
    # -----------------------------------------------------------------
    amount_t = _query_core(
        torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64),
    )

    # -----------------------------------------------------------------
    # 5️⃣ Build the profit sub‑query DataFrame
    # -----------------------------------------------------------------
    profit_df = cudf.DataFrame({
        "nation": joined["n_name"].reset_index(drop=True),
        "o_year": joined["o_orderdate"].dt.year.astype("int32").reset_index(drop=True),
        "amount": cudf.Series(amount_t),
    })

    # -----------------------------------------------------------------
    # 6️⃣ Group‑by aggregation and final ordering
    # -----------------------------------------------------------------
    grouped = profit_df.groupby(["nation", "o_year"], as_index=False).agg({"amount": "sum"})
    grouped = grouped.rename(columns={"amount": "sum_profit"})
    grouped = grouped.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)

    return cudf.DataFrame({
        "nation": grouped["nation"],
        "o_year": grouped["o_year"],
        "sum_profit": grouped["sum_profit"],
    })
