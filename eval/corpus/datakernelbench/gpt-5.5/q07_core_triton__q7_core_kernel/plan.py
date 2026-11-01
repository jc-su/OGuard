import os
import tempfile
import importlib.util
import linecache
from types import ModuleType
from typing import Any

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q7_core_kernel(
    supp_ptr,
    cust_ptr,
    shipdate_ptr,
    price_ptr,
    discount_ptr,
    out_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ship = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    keep_date = mask & (ship >= date_lo) & (ship <= date_hi)

    supp = tl.load(supp_ptr + offs, mask=keep_date, other=0)
    cust = tl.load(cust_ptr + offs, mask=keep_date, other=0)

    # Howard Hinnant civil_from_days year extraction, matching the Torch reference.
    z = ship + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    month = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(month <= 2, y + 1, y)
    year_off = year - 1995

    valid_year = (year_off >= 0) & (year_off < 2)
    keep = keep_date & valid_year

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    volume = price * (1.0 - disc)

    key = supp * 4 + cust * 2 + year_off

    s0 = tl.sum(tl.where(keep & (key == 0), volume, 0.0))
    s1 = tl.sum(tl.where(keep & (key == 1), volume, 0.0))
    s2 = tl.sum(tl.where(keep & (key == 2), volume, 0.0))
    s3 = tl.sum(tl.where(keep & (key == 3), volume, 0.0))
    s4 = tl.sum(tl.where(keep & (key == 4), volume, 0.0))
    s5 = tl.sum(tl.where(keep & (key == 5), volume, 0.0))
    s6 = tl.sum(tl.where(keep & (key == 6), volume, 0.0))
    s7 = tl.sum(tl.where(keep & (key == 7), volume, 0.0))

    tl.atomic_add(out_ptr + 0, s0, sem="relaxed")
    tl.atomic_add(out_ptr + 1, s1, sem="relaxed")
    tl.atomic_add(out_ptr + 2, s2, sem="relaxed")
    tl.atomic_add(out_ptr + 3, s3, sem="relaxed")
    tl.atomic_add(out_ptr + 4, s4, sem="relaxed")
    tl.atomic_add(out_ptr + 5, s5, sem="relaxed")
    tl.atomic_add(out_ptr + 6, s6, sem="relaxed")
    tl.atomic_add(out_ptr + 7, s7, sem="relaxed")
'''


def _load_kernels() -> ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q7_triton_")
    path = os.path.join(tmpdir, "_q7_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q7_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K: Any = _load_kernels()


def _launch_q7_core(
    supp_nation_code: torch.Tensor,
    cust_nation_code: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Runs on GPU with Triton: one fused pass over the joined tensor columns filters
    shipdate, computes year and volume, and atomically accumulates the eight tiny
    fixed groups. This avoids the eager Torch hot path's mask materialization,
    filtered column temporaries, volume tensor, key tensor, and scatter pass.
    """
    n = int(l_shipdate_days.numel())
    revenue = torch.zeros(8, dtype=torch.float64, device=l_extendedprice.device)
    if n == 0:
        return revenue
    grid = (triton.cdiv(n, block_size),)
    _K._q7_core_kernel[grid](
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        revenue,
        shipdate_lo_days,
        shipdate_hi_days,
        n,
        BLOCK_SIZE=block_size,
    )
    return revenue


def _query_core(
    supp_nation_code: torch.Tensor,
    cust_nation_code: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU tensor core for TPC-H Q7.

    Triton performs the memory-bound scan/reduce in a single fused kernel:
    filter by date, derive year, compute revenue, and aggregate into the fixed
    eight group slots. Tiny output shaping remains in Torch on eight elements,
    which is outside the scan hot path and preserves the reference return layout.
    """
    revenue = _launch_q7_core(
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    group_ids = torch.arange(8, dtype=torch.int64, device=revenue.device)
    out_supp = (group_ids // 4).to(torch.int32)
    out_cust = ((group_ids // 2) % 2).to(torch.int32)
    out_year = (group_ids % 2).to(torch.int32) + 1995

    valid = ((out_supp == 0) & (out_cust == 1)) | ((out_supp == 1) & (out_cust == 0))
    valid = valid & (revenue > 0)

    return out_supp[valid], out_cust[valid], out_year[valid], revenue[valid]


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_a: str = "FRANCE",
    nation_b: str = "GERMANY",
    shipdate_lo: str = "1995-01-01",
    shipdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL: SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue FROM (
         SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
                extract(year FROM l_shipdate) AS l_year,
                l_extendedprice * (1 - l_discount) AS volume
         FROM supplier, lineitem, orders, customer, nation n1, nation n2
         WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey
               AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey
               AND c_nationkey = n2.n_nationkey
               AND ((n1.n_name='FRANCE' AND n2.n_name='GERMANY')
                 OR (n1.n_name='GERMANY' AND n2.n_name='FRANCE'))
               AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
         ) GROUP BY supp_nation, cust_nation, l_year
         ORDER BY supp_nation, cust_nation, l_year;
    Description: TPC-H Q7 - Shipping revenue between two nation pairs by year.
    Params:
      nation_a (str): first nation name.
      nation_b (str): second nation name.
      shipdate_lo (str): inclusive lower bound for l_shipdate.
      shipdate_hi (str): inclusive upper bound for l_shipdate.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]

    # Restrict nations to only the two relevant ones
    nations2 = df_nation[df_nation["n_name"].isin([nation_a, nation_b])][
        ["n_nationkey", "n_name"]
    ]

    # Join supplier with nation (supp_nation)
    sup = df_supplier[["s_suppkey", "s_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "s_nationkey", "n_name": "supp_nation"}),
        on="s_nationkey",
    )

    # Join customer with nation (cust_nation)
    cust = df_customer[["c_custkey", "c_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "c_nationkey", "n_name": "cust_nation"}),
        on="c_nationkey",
    )

    # lineitem with shipdate filter
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    li = df_lineitem[["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]]
    li = li.merge(sup[["s_suppkey", "supp_nation"]], left_on="l_suppkey", right_on="s_suppkey")

    # orders join customer
    oc = df_orders[["o_orderkey", "o_custkey"]].merge(
        cust[["c_custkey", "cust_nation"]], left_on="o_custkey", right_on="c_custkey"
    )

    joined = li.merge(oc[["o_orderkey", "cust_nation"]], left_on="l_orderkey", right_on="o_orderkey")

    # Keep only (FRANCE,GERMANY) or (GERMANY,FRANCE) pairs
    pair_mask = (
        ((joined["supp_nation"] == nation_a) & (joined["cust_nation"] == nation_b))
        | ((joined["supp_nation"] == nation_b) & (joined["cust_nation"] == nation_a))
    )
    joined = joined[pair_mask]

    # Encode nation names: nation_a -> 0, nation_b -> 1
    supp_code = (joined["supp_nation"] == nation_b).astype("int32")
    cust_code = (joined["cust_nation"] == nation_b).astype("int32")

    l_shipdate_days = ((joined["l_shipdate"].astype("int64") // 86400).astype("int32"))

    supp_t = torch.as_tensor(supp_code.to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(cust_code.to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    supp_out, cust_out, year_out, rev_out = _query_core(
        supp_t, cust_t, ship_t, price_t, disc_t, shipdate_lo_days, shipdate_hi_days
    )

    # Map codes back to names: 0 -> nation_a, 1 -> nation_b
    supp_out_cpu = supp_out.cpu().numpy()
    cust_out_cpu = cust_out.cpu().numpy()
    supp_names = [nation_a if v == 0 else nation_b for v in supp_out_cpu]
    cust_names = [nation_a if v == 0 else nation_b for v in cust_out_cpu]

    result = cudf.DataFrame({
        "supp_nation": cudf.Series(supp_names),
        "cust_nation": cudf.Series(cust_names),
        "l_year": year_out,
        "revenue": rev_out,
    })

    result = result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
    return result
