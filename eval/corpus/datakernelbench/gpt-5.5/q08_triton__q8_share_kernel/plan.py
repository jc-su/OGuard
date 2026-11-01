import os
import tempfile
import importlib.util
import linecache
from typing import Any

import cudf
import pandas as pd
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q8_agg_kernel(
    year_ptr,
    price_ptr,
    discount_ptr,
    target_flag_ptr,
    total_ptr,
    target_ptr,
    n_elements: tl.constexpr,
    base_year: tl.constexpr,
    NUM_YEARS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_elements

    year = tl.load(year_ptr + offs, mask=row_mask, other=base_year - 1)
    idx = year - base_year
    in_range = row_mask & (idx >= 0) & (idx < NUM_YEARS)

    price = tl.load(price_ptr + offs, mask=in_range, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=in_range, other=0.0)
    flag = tl.load(target_flag_ptr + offs, mask=in_range, other=0)

    volume = price * (1.0 - discount)
    is_target = flag != 0

    for g in tl.static_range(0, NUM_YEARS):
        keep_g = in_range & (idx == g)
        total_sum = tl.sum(tl.where(keep_g, volume, 0.0))
        target_sum = tl.sum(tl.where(keep_g & is_target, volume, 0.0))
        tl.atomic_add(total_ptr + g, total_sum, sem="relaxed")
        tl.atomic_add(target_ptr + g, target_sum, sem="relaxed")


@triton.jit
def _q8_share_kernel(
    target_ptr,
    total_ptr,
    share_ptr,
    n_years,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_years

    total = tl.load(total_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)
    safe_total = tl.where(total != 0.0, total, 1.0)
    share = target / safe_total

    tl.store(share_ptr + offs, share, mask=mask)
'''


def _load_kernels() -> Any:
    """Write Triton JIT kernels to a real file and import them for validator-safe compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q8_triton_")
    path = os.path.join(tmpdir, "_q8_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q8_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _empty_result() -> cudf.DataFrame:
    """Create an empty cuDF result with the exact output schema."""
    return cudf.DataFrame(
        {
            "o_year": cudf.Series([], dtype="int32"),
            "mkt_share": cudf.Series([], dtype="float64"),
        }
    )


def _share_block_size(n_years: int) -> int:
    """Choose a small power-of-two block for the tiny year-share finalization kernel."""
    if n_years <= 16:
        return 16
    bs = 1 << (n_years - 1).bit_length()
    return min(bs, 1024)


def _launch_q8_aggregate(
    o_year: torch.Tensor,
    extendedprice: torch.Tensor,
    discount: torch.Tensor,
    is_target_nation: torch.Tensor,
    base_year: int,
    num_years: int,
    block_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the GPU hot path in Triton.

    The kernel performs one fused pass over joined lineitem rows: load year/price/discount/target flag,
    compute volume in registers, and atomically accumulate total and target-nation volume into the
    small fixed set of year bins. This avoids PyTorch temporary tensors, torch.unique, scatter_add,
    and multiple HBM passes in the reference.
    """
    n = int(extendedprice.numel())
    total = torch.zeros(num_years, dtype=torch.float64, device=extendedprice.device)
    target = torch.zeros(num_years, dtype=torch.float64, device=extendedprice.device)
    share = torch.empty(num_years, dtype=torch.float64, device=extendedprice.device)

    if n == 0 or num_years <= 0:
        return total, share

    grid = (triton.cdiv(n, block_size),)
    _K._q8_agg_kernel[grid](
        o_year,
        extendedprice,
        discount,
        is_target_nation,
        total,
        target,
        n,
        base_year,
        num_years,
        BLOCK_SIZE=block_size,
    )

    share_bs = _share_block_size(num_years)
    share_grid = (triton.cdiv(num_years, share_bs),)
    _K._q8_share_kernel[share_grid](
        target,
        total,
        share,
        num_years,
        BLOCK_SIZE=share_bs,
    )
    return total, share


def _query_core(
    o_year: torch.Tensor,
    extendedprice: torch.Tensor,
    discount: torch.Tensor,
    is_target_nation: torch.Tensor,
    base_year: int,
    num_years: int,
    block_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate rows on GPU with Triton and return sorted non-empty year groups.

    Triton handles the memory-bound scan/project/reduce. A tiny PyTorch post-step only filters the
    already aggregated year bins, preserving the reference result shape and ascending year order.
    """
    total, share = _launch_q8_aggregate(
        o_year=o_year,
        extendedprice=extendedprice,
        discount=discount,
        is_target_nation=is_target_nation,
        base_year=base_year,
        num_years=num_years,
        block_size=block_size,
    )

    present = total != 0.0
    years = (
        torch.arange(num_years, dtype=torch.int32, device=extendedprice.device)
        + int(base_year)
    )
    return years[present], share[present]


def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "AMERICA",
    part_type: str = "ECONOMY ANODIZED STEEL",
    nation_name: str = "BRAZIL",
    orderdate_lo: str = "1995-01-01",
    orderdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT
            o_year,
            sum(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / sum(volume) AS mkt_share
        FROM (
            SELECT
                extract(year FROM o_orderdate) AS o_year,
                l_extendedprice * (1 - l_discount) AS volume,
                n2.n_name AS nation
            FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
            WHERE p_partkey = l_partkey
              AND s_suppkey = l_suppkey
              AND l_orderkey = o_orderkey
              AND o_custkey = c_custkey
              AND c_nationkey = n1.n_nationkey
              AND n1.n_regionkey = r_regionkey
              AND r_name = 'AMERICA'
              AND s_nationkey = n2.n_nationkey
              AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
              AND p_type = 'ECONOMY ANODIZED STEEL'
        ) AS all_nations
        GROUP BY o_year
        ORDER BY o_year;

    Description:
        TPC-H Q8 market share by order year. cuDF performs string predicates, key filtering, and joins
        where hash joins and string columns are already efficient. The bulk numeric hot path after the
        joins runs in Triton: one fused GPU pass computes volume and accumulates total/target volume
        per year, reducing HBM traffic and avoiding PyTorch unique/scatter temporaries.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables keyed by table name.
        region_name (str): Customer-region filter.
        part_type (str): Part type filter.
        nation_name (str): Supplier nation whose market share is computed.
        orderdate_lo (str): Inclusive lower order-date bound.
        orderdate_hi (str): Inclusive upper order-date bound.
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    if hi_days < lo_days:
        return _empty_result()

    region_keys = df_region.loc[df_region["r_name"] == region_name, "r_regionkey"]
    if len(region_keys) == 0:
        return _empty_result()

    customer_nation_keys = df_nation.loc[
        df_nation["n_regionkey"].isin(region_keys), "n_nationkey"
    ]
    if len(customer_nation_keys) == 0:
        return _empty_result()

    customer_keys = df_customer.loc[
        df_customer["c_nationkey"].isin(customer_nation_keys), "c_custkey"
    ]
    if len(customer_keys) == 0:
        return _empty_result()

    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_mask = (
        (o_days >= lo_days)
        & (o_days <= hi_days)
        & df_orders["o_custkey"].isin(customer_keys)
    )
    orders_f = df_orders.loc[orders_mask, ["o_orderkey", "o_orderdate"]]
    if len(orders_f) == 0:
        return _empty_result()

    part_keys = df_part.loc[df_part["p_type"] == part_type, "p_partkey"]
    if len(part_keys) == 0:
        return _empty_result()

    li = df_lineitem.loc[
        df_lineitem["l_partkey"].isin(part_keys),
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"],
    ]
    if len(li) == 0:
        return _empty_result()

    li = li.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[
        ["l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"]
    ]
    if len(li) == 0:
        return _empty_result()

    target_nation_keys = df_nation.loc[df_nation["n_name"] == nation_name, "n_nationkey"]
    target_supplier_keys = df_supplier.loc[
        df_supplier["s_nationkey"].isin(target_nation_keys), "s_suppkey"
    ]

    is_target_series = li["l_suppkey"].isin(target_supplier_keys).astype("int8")
    o_year_series = li["o_orderdate"].dt.year.astype("int32")

    base_year = int(o_year_series.min())
    max_year = int(o_year_series.max())
    num_years = max_year - base_year + 1
    if num_years <= 0:
        return _empty_result()

    o_year_t = torch.as_tensor(o_year_series.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64)
    is_target_t = torch.as_tensor(is_target_series.to_cupy(), dtype=torch.int8)

    years_t, mkt_share_t = _query_core(
        o_year=o_year_t,
        extendedprice=price_t,
        discount=discount_t,
        is_target_nation=is_target_t,
        base_year=base_year,
        num_years=num_years,
    )

    return cudf.DataFrame(
        {
            "o_year": years_t,
            "mkt_share": mkt_share_t,
        }
    )
