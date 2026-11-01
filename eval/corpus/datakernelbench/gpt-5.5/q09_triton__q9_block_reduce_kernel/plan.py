import os
import tempfile
import importlib.util
import linecache
import types
from typing import Any

import cudf
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q9_block_reduce_kernel(
    extendedprice_ptr,
    discount_ptr,
    supplycost_ptr,
    quantity_ptr,
    nationkey_ptr,
    year_ptr,
    sum_out_ptr,
    cnt_out_ptr,
    n_elements,
    min_year,
    N_YEARS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nation = tl.load(nationkey_ptr + offs, mask=mask, other=-1)
    year = tl.load(year_ptr + offs, mask=mask, other=-2147483648)
    yoff = year - min_year

    price = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    cost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    amount = price * (1.0 - disc) - cost * qty
    valid = mask & (nation >= 0) & (nation < 25) & (yoff >= 0) & (yoff < N_YEARS)

    for ni in tl.static_range(0, 25):
        for yi in tl.static_range(0, 16):
            if yi < N_YEARS:
                gmask = valid & (nation == ni) & (yoff == yi)
                partial_sum = tl.sum(tl.where(gmask, amount, 0.0), axis=0)
                partial_cnt = tl.sum(tl.where(gmask, 1.0, 0.0), axis=0)
                out_idx = ni * N_YEARS + yi
                has_rows = partial_cnt > 0.0
                tl.atomic_add(sum_out_ptr + out_idx, partial_sum, sem="relaxed", mask=has_rows)
                tl.atomic_add(cnt_out_ptr + out_idx, partial_cnt, sem="relaxed", mask=has_rows)


@triton.jit
def _q9_row_atomic_kernel(
    extendedprice_ptr,
    discount_ptr,
    supplycost_ptr,
    quantity_ptr,
    nationkey_ptr,
    year_ptr,
    sum_out_ptr,
    cnt_out_ptr,
    n_elements,
    min_year,
    n_years,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nation = tl.load(nationkey_ptr + offs, mask=mask, other=-1)
    year = tl.load(year_ptr + offs, mask=mask, other=-2147483648)
    yoff = year - min_year
    valid = mask & (nation >= 0) & (nation < 25) & (yoff >= 0) & (yoff < n_years)

    price = tl.load(extendedprice_ptr + offs, mask=valid, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=valid, other=0.0)
    cost = tl.load(supplycost_ptr + offs, mask=valid, other=0.0)
    qty = tl.load(quantity_ptr + offs, mask=valid, other=0.0)

    amount = price * (1.0 - disc) - cost * qty
    out_idx = nation * n_years + yoff
    tl.atomic_add(sum_out_ptr + out_idx, amount, sem="relaxed", mask=valid)
    tl.atomic_add(cnt_out_ptr + out_idx, tl.full((BLOCK_SIZE,), 1.0, tl.float64), sem="relaxed", mask=valid)
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q9_triton_")
    path = os.path.join(tmpdir, "_q9_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q9_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import specification for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _empty_result(df_nation: cudf.DataFrame) -> cudf.DataFrame:
    """Create an empty cuDF result with the exact output column order and dtypes."""
    return cudf.DataFrame(
        {
            "nation": cudf.Series([], dtype=df_nation["n_name"].dtype),
            "o_year": cudf.Series([], dtype="int32"),
            "sum_profit": cudf.Series([], dtype="float64"),
        }
    )


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
    s_nationkey: torch.Tensor,
    o_year: torch.Tensor,
    min_year: int,
    n_years: int,
    block_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the hot projection and small-group aggregation on the GPU with Triton.

    The input rows are already joined by cuDF.  This Triton path fuses the Q9 amount
    expression with aggregation into fixed nation/year bins, avoiding materializing an
    intermediate amount column and avoiding a cuDF groupby over the joined fact rows.
    For the normal TPC-H year span, each program reduces a block locally per group and
    emits only one atomic update per non-empty group in the block.
    """
    n = int(l_extendedprice.numel())
    device = l_extendedprice.device
    n_groups = 25 * int(n_years)
    sum_out = torch.zeros(n_groups, dtype=torch.float64, device=device)
    cnt_out = torch.zeros(n_groups, dtype=torch.float64, device=device)
    if n == 0 or n_years <= 0:
        return sum_out, cnt_out

    if n_years <= 16:
        grid = (triton.cdiv(n, block_size),)
        _K._q9_block_reduce_kernel[grid](
            l_extendedprice,
            l_discount,
            ps_supplycost,
            l_quantity,
            s_nationkey,
            o_year,
            sum_out,
            cnt_out,
            n,
            int(min_year),
            N_YEARS=int(n_years),
            BLOCK_SIZE=int(block_size),
            num_warps=8,
        )
    else:
        row_block = 256
        grid = (triton.cdiv(n, row_block),)
        _K._q9_row_atomic_kernel[grid](
            l_extendedprice,
            l_discount,
            ps_supplycost,
            l_quantity,
            s_nationkey,
            o_year,
            sum_out,
            cnt_out,
            n,
            int(min_year),
            int(n_years),
            BLOCK_SIZE=row_block,
            num_warps=4,
        )
    return sum_out, cnt_out


def _build_result_frame(
    sum_profit: torch.Tensor,
    counts: torch.Tensor,
    min_year: int,
    n_years: int,
    df_nation: cudf.DataFrame,
) -> cudf.DataFrame:
    """
    Build the final small cuDF result from Triton aggregate buffers.

    Triton returns dense aggregate slots keyed by nationkey and year offset.  cuDF is
    used only for the small metadata join to nation names, filtering empty groups, and
    final SQL ordering.
    """
    nation_keys: list[int] = []
    years: list[int] = []
    for nk in range(25):
        for yoff in range(int(n_years)):
            nation_keys.append(nk)
            years.append(int(min_year) + yoff)

    grouped = cudf.DataFrame(
        {
            "s_nationkey": cudf.Series(nation_keys, dtype="int64"),
            "o_year": cudf.Series(years, dtype="int32"),
            "sum_profit": cudf.Series(sum_profit),
            "_cnt": cudf.Series(counts),
        }
    )
    grouped = grouped[grouped["_cnt"] > 0.0]
    if len(grouped) == 0:
        return _empty_result(df_nation)

    nation_dim = df_nation[["n_nationkey", "n_name"]]
    out = grouped.merge(
        nation_dim,
        left_on="s_nationkey",
        right_on="n_nationkey",
        how="inner",
    )
    if len(out) == 0:
        return _empty_result(df_nation)

    out = out.rename(columns={"n_name": "nation"})
    out = out[["nation", "o_year", "sum_profit"]]
    out = out.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)
    return cudf.DataFrame(
        {
            "nation": out["nation"],
            "o_year": out["o_year"].astype("int32"),
            "sum_profit": out["sum_profit"].astype("float64"),
        }
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    name_pattern: str = "green",
    block_size: int = 512,
) -> cudf.DataFrame:
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
              AND p_name LIKE '%green%'
        ) AS profit
        GROUP BY nation, o_year
        ORDER BY nation, o_year DESC;

    Description:
        TPC-H Q9 product type profit.  cuDF performs string filtering and relational
        joins, while the GPU hot path after the joins is a fused Triton projection plus
        nation/year aggregation.  This is faster than the reference because it avoids
        building a full intermediate amount column and replaces a large cuDF groupby
        with one coalesced scan and per-block reductions into the small fixed group set.

    Params:
        tables: Dictionary of cuDF input tables.
        name_pattern: Substring to match in p_name; percent wildcards are stripped to
            preserve the reference behavior.
        block_size: Triton tile size for the fused reduction kernel.
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    pat = name_pattern.replace("%", "")

    part_keys = df_part[df_part["p_name"].str.contains(pat)]["p_partkey"]
    if len(part_keys) == 0:
        return _empty_result(df_nation)

    li_base = df_lineitem[
        [
            "l_partkey",
            "l_suppkey",
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_quantity",
        ]
    ]
    li_filtered = li_base[li_base["l_partkey"].isin(part_keys)]
    if len(li_filtered) == 0:
        return _empty_result(df_nation)

    ps_base = df_partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]]
    ps_filtered = ps_base[ps_base["ps_partkey"].isin(part_keys)]
    if len(ps_filtered) == 0:
        return _empty_result(df_nation)

    li_ps = li_filtered.merge(
        ps_filtered,
        left_on=["l_partkey", "l_suppkey"],
        right_on=["ps_partkey", "ps_suppkey"],
        how="inner",
    )
    if len(li_ps) == 0:
        return _empty_result(df_nation)

    li_ps = li_ps[
        [
            "l_suppkey",
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_quantity",
            "ps_supplycost",
        ]
    ]

    supplier_dim = df_supplier[["s_suppkey", "s_nationkey"]]
    li_ps_s = li_ps.merge(
        supplier_dim,
        left_on="l_suppkey",
        right_on="s_suppkey",
        how="inner",
    )
    if len(li_ps_s) == 0:
        return _empty_result(df_nation)

    li_ps_s = li_ps_s[
        [
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_quantity",
            "ps_supplycost",
            "s_nationkey",
        ]
    ]

    orders_year = cudf.DataFrame(
        {
            "o_orderkey": df_orders["o_orderkey"],
            "o_year": df_orders["o_orderdate"].dt.year.astype("int32"),
        }
    )
    joined = li_ps_s.merge(
        orders_year,
        left_on="l_orderkey",
        right_on="o_orderkey",
        how="inner",
    )
    if len(joined) == 0:
        return _empty_result(df_nation)

    joined = joined[
        [
            "l_extendedprice",
            "l_discount",
            "ps_supplycost",
            "l_quantity",
            "s_nationkey",
            "o_year",
        ]
    ]

    min_year = int(joined["o_year"].min())
    max_year = int(joined["o_year"].max())
    n_years = max_year - min_year + 1
    if n_years <= 0:
        return _empty_result(df_nation)

    l_extendedprice = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    ps_supplycost = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    l_quantity = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    s_nationkey = torch.as_tensor(joined["s_nationkey"].to_cupy(), dtype=torch.int64)
    o_year = torch.as_tensor(joined["o_year"].to_cupy(), dtype=torch.int32)

    sum_profit, counts = _query_core(
        l_extendedprice,
        l_discount,
        ps_supplycost,
        l_quantity,
        s_nationkey,
        o_year,
        min_year,
        n_years,
        block_size=block_size,
    )

    return _build_result_frame(sum_profit, counts, min_year, n_years, df_nation)
