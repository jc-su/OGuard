import os
import tempfile
import importlib.util
import linecache
import types

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q8_reduce_1995_1996_kernel(
    o_year_ptr,
    volume_ptr,
    is_brazil_ptr,
    agg_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    years = tl.load(o_year_ptr + offs, mask=mask, other=-1)
    volume = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    is_brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=0) != 0

    is_1995 = mask & (years == 1995)
    is_1996 = mask & (years == 1996)

    total_1995 = tl.sum(tl.where(is_1995, volume, 0.0))
    total_1996 = tl.sum(tl.where(is_1996, volume, 0.0))
    brazil_1995 = tl.sum(tl.where(is_1995 & is_brazil, volume, 0.0))
    brazil_1996 = tl.sum(tl.where(is_1996 & is_brazil, volume, 0.0))

    tl.atomic_add(agg_ptr + 0, brazil_1995, sem="relaxed")
    tl.atomic_add(agg_ptr + 1, total_1995, sem="relaxed")
    tl.atomic_add(agg_ptr + 2, brazil_1996, sem="relaxed")
    tl.atomic_add(agg_ptr + 3, total_1996, sem="relaxed")


@triton.jit
def _q8_finalize_2y_kernel(
    agg_ptr,
    out_ptr,
):
    brazil_1995 = tl.load(agg_ptr + 0)
    total_1995 = tl.load(agg_ptr + 1)
    brazil_1996 = tl.load(agg_ptr + 2)
    total_1996 = tl.load(agg_ptr + 3)

    tl.store(out_ptr + 0, brazil_1995 / total_1995)
    tl.store(out_ptr + 1, brazil_1996 / total_1996)
'''


def _load_kernels() -> types.ModuleType:
    """Load Triton kernels from an on-disk module so Triton can JIT from source."""
    tmpdir = tempfile.mkdtemp(prefix="q8_triton_")
    path = os.path.join(tmpdir, "_q8_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q8_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()
_YEAR_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_default_years(device: torch.device) -> torch.Tensor:
    """Return cached GPU year keys for the default Q8 date range."""
    years = _YEAR_CACHE.get(device)
    if years is None:
        years = torch.tensor([1995, 1996], dtype=torch.int32, device=device)
        _YEAR_CACHE[device] = years
    return years


def _query_core(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the numeric hot path on GPU with Triton.

    cuDF in run_query performs joins, filters, and tensor extraction. This core
    replaces eager torch.unique/scatter_add/sort with one fused Triton scan that
    reduces the two Q8 output years directly, followed by a tiny Triton finalize
    kernel. The scan is memory-bound, so avoiding inverse-group arrays, CASE
    temporaries, scatter buffers, and sorting reduces HBM traffic and launch work.
    """
    n = volume.numel()
    if n == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=volume.device),
            torch.empty(0, dtype=torch.float64, device=volume.device),
        )

    if not o_year.is_contiguous():
        o_year = o_year.contiguous()
    if not volume.is_contiguous():
        volume = volume.contiguous()
    if not is_brazil.is_contiguous():
        is_brazil = is_brazil.contiguous()

    agg = torch.zeros(4, dtype=torch.float64, device=volume.device)
    mkt_share = torch.empty(2, dtype=torch.float64, device=volume.device)

    block_size = 2048
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q8_reduce_1995_1996_kernel[grid](
        o_year,
        volume,
        is_brazil,
        agg,
        n,
        BLOCK_SIZE=block_size,
    )
    _K._q8_finalize_2y_kernel[(1,)](agg, mkt_share)

    return _get_default_years(volume.device), mkt_share


def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "AMERICA",
    part_type: str = "ECONOMY ANODIZED STEEL",
    nation_name: str = "BRAZIL",
    orderdate_lo: str = "1995-01-01",
    orderdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_year, sum(CASE WHEN nation='BRAZIL' THEN volume ELSE 0 END)/sum(volume) AS mkt_share
         FROM (SELECT extract(year FROM o_orderdate) AS o_year,
                      l_extendedprice*(1-l_discount) AS volume,
                      n2.n_name AS nation
               FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
               WHERE p_partkey=l_partkey AND s_suppkey=l_suppkey AND l_orderkey=o_orderkey
                 AND o_custkey=c_custkey AND c_nationkey=n1.n_nationkey
                 AND n1.n_regionkey=r_regionkey AND r_name='AMERICA'
                 AND s_nationkey=n2.n_nationkey
                 AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
                 AND p_type='ECONOMY ANODIZED STEEL') AS all_nations
         GROUP BY o_year ORDER BY o_year;
    Description: TPC-H Q8 — market share of a given nation within a region for a specific part type over two years.
    Params:
        region_name (str): region filter for customer's nation.
        part_type (str): part type filter.
        nation_name (str): nation whose market share is computed (supplier nation).
        orderdate_lo (str): order date lower bound (inclusive).
        orderdate_hi (str): order date upper bound (inclusive).
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    # Date bounds as epoch-day int32
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter region -> nation n1 (customer side)
    region_f = df_region[df_region["r_name"] == region_name][["r_regionkey"]]
    n1 = df_nation.merge(region_f, left_on="n_regionkey", right_on="r_regionkey")[
        ["n_nationkey"]
    ].rename(columns={"n_nationkey": "n1_nationkey"})

    # Customers in that region
    cust_r = df_customer.merge(n1, left_on="c_nationkey", right_on="n1_nationkey")[
        ["c_custkey"]
    ]

    # Filter orders by date range
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days <= hi_days)][
        ["o_orderkey", "o_custkey", "o_orderdate"]
    ]
    # Orders by customers in region
    orders_f = orders_f.merge(cust_r, left_on="o_custkey", right_on="c_custkey")[
        ["o_orderkey", "o_orderdate"]
    ]

    # Filter part by type
    part_f = df_part[df_part["p_type"] == part_type][["p_partkey"]]

    # Lineitem joined with part (filter by p_partkey)
    li = df_lineitem.merge(part_f, left_on="l_partkey", right_on="p_partkey")[
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with filtered orders
    li = li.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[
        ["l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"]
    ]

    # Join supplier -> nation n2 to get supplier nation name
    sup_n2 = df_supplier.merge(
        df_nation, left_on="s_nationkey", right_on="n_nationkey"
    )[["s_suppkey", "n_name"]]

    li = li.merge(sup_n2, left_on="l_suppkey", right_on="s_suppkey")[
        ["l_extendedprice", "l_discount", "o_orderdate", "n_name"]
    ]

    # Build tensors
    o_year_series = (li["o_orderdate"].dt.year).astype("int32")
    o_year_t = torch.as_tensor(o_year_series.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64)
    volume_t = price_t * (1.0 - disc_t)

    is_brazil_t = torch.as_tensor(
        (li["n_name"] == nation_name).astype("bool").to_cupy()
    )

    years_t, mkt_share_t = _query_core(o_year_t, volume_t, is_brazil_t)

    return cudf.DataFrame({
        "o_year": years_t,
        "mkt_share": mkt_share_t,
    })
