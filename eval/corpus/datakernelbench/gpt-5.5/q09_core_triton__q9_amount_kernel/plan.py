import os
import tempfile
import importlib.util
import linecache
import types

import torch
import triton
import cudf


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q9_amount_kernel(
    l_extendedprice_ptr,
    l_discount_ptr,
    ps_supplycost_ptr,
    l_quantity_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    extendedprice = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    supplycost = tl.load(ps_supplycost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)

    one = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    amount = extendedprice * (one - discount) - supplycost * quantity
    tl.store(out_ptr + offs, amount, mask=mask)
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation.

    Runs on CPU at module import time. Triton requires kernels to come from an
    on-disk source file under this validation environment, so the GPU hot path
    remains in `_KERNEL_SOURCE` while host-side cuDF/Torch orchestration stays in
    this module.
    """
    tmpdir = tempfile.mkdtemp(prefix="q9_triton_")
    path = os.path.join(tmpdir, "_q9_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)

    spec = importlib.util.spec_from_file_location("_q9_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import spec for Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q9_amount(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Launch the fused Triton amount projection on GPU.

    The reference eager Torch expression materializes intermediate tensors for
    `(1 - discount)`, `extendedprice * ...`, and `supplycost * quantity`.
    This launcher performs the same per-row arithmetic in one coalesced Triton
    pass over four input columns and writes one output column, reducing kernel
    launches and temporary HBM traffic for this memory-bound projection.
    """
    n = l_extendedprice.numel()
    out = torch.empty_like(l_extendedprice)
    if n == 0:
        return out

    grid = (triton.cdiv(n, block_size),)
    _K._q9_amount_kernel[grid](
        l_extendedprice,
        l_discount,
        ps_supplycost,
        l_quantity,
        out,
        n,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """Compute Q9 row profit amounts with a fused Triton GPU kernel.

    Inputs are aligned 1D GPU tensors produced by `run_query` after cuDF joins.
    The numeric hot path runs on GPU in Triton, replacing eager PyTorch
    elementwise kernels with one fused pass:
    `extendedprice * (1 - discount) - supplycost * quantity`.
    """
    return _launch_q9_amount(l_extendedprice, l_discount, ps_supplycost, l_quantity)


def run_query(tables: dict[str, cudf.DataFrame], name_pattern: str = "green") -> cudf.DataFrame:
    """
    SQL: SELECT nation, o_year, sum(amount) AS sum_profit FROM (SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM part, supplier, lineitem, partsupp, orders, nation WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey AND p_name LIKE '%green%') AS profit GROUP BY nation, o_year ORDER BY nation, o_year DESC;
    Description: TPC-H Q9 - Product Type Profit Measure. Computes profit by nation and year for parts whose name matches a pattern.
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

    # Compute amount via torch core
    amount_t = _query_core(
        torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64),
    )

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
