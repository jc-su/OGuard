import os
import tempfile
import importlib.util
import linecache
import types
from typing import Any

import cudf
import pandas as pd
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q7_agg_kernel(
    supp_code_ptr,
    cust_code_ptr,
    shipdate_days_ptr,
    price_ptr,
    discount_ptr,
    date_lo,
    date_hi,
    year_split_days,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    supp = tl.load(supp_code_ptr + offs, mask=mask, other=0)
    cust = tl.load(cust_code_ptr + offs, mask=mask, other=0)
    ship = tl.load(shipdate_days_ptr + offs, mask=mask, other=0)

    pair_keep = ((supp == 0) & (cust == 1)) | ((supp == 1) & (cust == 0))
    date_keep = (ship >= date_lo) & (ship <= date_hi)
    keep = mask & pair_keep & date_keep

    year_offset = tl.where(ship < year_split_days, 0, 1)
    key = supp * 4 + cust * 2 + year_offset

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    volume = price * (1.0 - discount)

    s2 = tl.sum(tl.where(key == 2, volume, 0.0))
    s3 = tl.sum(tl.where(key == 3, volume, 0.0))
    s4 = tl.sum(tl.where(key == 4, volume, 0.0))
    s5 = tl.sum(tl.where(key == 5, volume, 0.0))

    tl.atomic_add(out_ptr + 2, s2)
    tl.atomic_add(out_ptr + 3, s3)
    tl.atomic_add(out_ptr + 4, s4)
    tl.atomic_add(out_ptr + 5, s5)
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q7_triton_")
    path = os.path.join(tmpdir, "_q7_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q7_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import spec for Triton Q7 kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _empty_result() -> cudf.DataFrame:
    """Create an empty cuDF result with the SQL output columns and expected dtypes."""
    return cudf.DataFrame(
        {
            "supp_nation": cudf.Series([], dtype="str"),
            "cust_nation": cudf.Series([], dtype="str"),
            "l_year": cudf.Series([], dtype="int32"),
            "revenue": cudf.Series([], dtype="float64"),
        }
    )


def _epoch_days(date_text: str) -> int:
    """Convert a date string on the host to int32-compatible epoch days, matching the reference."""
    return int(pd.to_datetime(date_text).timestamp() // 86400)


def _launch_q7_agg(
    supp_code: torch.Tensor,
    cust_code: torch.Tensor,
    shipdate_days: torch.Tensor,
    extendedprice: torch.Tensor,
    discount: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    year_split_days: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Run the fused Triton GPU aggregation.

    The kernel performs the remaining pair/date predicates, year projection, volume computation,
    and four fixed-group FP64 reductions in one coalesced pass over the joined columns. This avoids
    the reference Torch mask materialization, filtered gathers, volume temporary, and scatter_add
    hot path.
    """
    n = int(extendedprice.numel())
    out = torch.zeros(8, dtype=torch.float64, device=extendedprice.device)
    if n == 0:
        return out

    grid = (triton.cdiv(n, block_size),)
    _K._q7_agg_kernel[grid](
        supp_code,
        cust_code,
        shipdate_days,
        extendedprice,
        discount,
        shipdate_lo_days,
        shipdate_hi_days,
        year_split_days,
        out,
        n,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


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
    GPU-only core compatible with the reference shape.

    Triton does the memory-bound filter/project/aggregate in one kernel. Tiny output expansion is
    done with Torch tensors because only up to four groups are produced.
    """
    year_split_days = _epoch_days("1996-01-01")
    revenue = _launch_q7_agg(
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        shipdate_lo_days,
        shipdate_hi_days,
        year_split_days,
    )

    group_ids = torch.arange(8, dtype=torch.int64, device=revenue.device)
    out_supp = (group_ids // 4).to(torch.int32)
    out_cust = ((group_ids // 2) % 2).to(torch.int32)
    out_year = (group_ids % 2).to(torch.int32) + 1995
    valid = (((out_supp == 0) & (out_cust == 1)) | ((out_supp == 1) & (out_cust == 0))) & (
        revenue > 0
    )
    return out_supp[valid], out_cust[valid], out_year[valid], revenue[valid]


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_a: str = "FRANCE",
    nation_b: str = "GERMANY",
    shipdate_lo: str = "1995-01-01",
    shipdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue
        FROM (
            SELECT n1.n_name AS supp_nation,
                   n2.n_name AS cust_nation,
                   extract(year FROM l_shipdate) AS l_year,
                   l_extendedprice * (1 - l_discount) AS volume
            FROM supplier, lineitem, orders, customer, nation n1, nation n2
            WHERE s_suppkey = l_suppkey
              AND o_orderkey = l_orderkey
              AND c_custkey = o_custkey
              AND s_nationkey = n1.n_nationkey
              AND c_nationkey = n2.n_nationkey
              AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
                OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
              AND l_shipdate BETWEEN CAST('1995-01-01' AS date)
                                  AND CAST('1996-12-31' AS date)
        ) AS shipping
        GROUP BY supp_nation, cust_nation, l_year
        ORDER BY supp_nation, cust_nation, l_year;

    Description:
        TPC-H Q7 shipping revenue between two nation pairs by year. cuDF handles the relational
        work that benefits from hash joins and string processing: tiny nation lookup, supplier and
        customer filtering, orders/customer join, date predicate pushdown, and the reduced lineitem
        joins. The hot numeric tail is a fused Triton GPU kernel over the joined numeric columns:
        it applies the remaining pair/date predicates, computes l_year and volume in registers, and
        atomically accumulates the four possible FP64 groups. This is faster than the reference
        Torch plan because it avoids multiple eager tensor masks, filtered temporary columns, a
        separate volume tensor, and scatter_add over materialized rows.

    Params:
        tables (dict[str, cudf.DataFrame]): TPC-H tables keyed by table name.
        nation_a (str): first nation name; encoded as code 0.
        nation_b (str): second nation name; encoded as code 1.
        shipdate_lo (str): inclusive lower bound for l_shipdate.
        shipdate_hi (str): inclusive upper bound for l_shipdate.
    """
    if nation_a == nation_b:
        return _empty_result()

    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]

    shipdate_lo_days = _epoch_days(shipdate_lo)
    shipdate_hi_days = _epoch_days(shipdate_hi)
    year_split_days = _epoch_days("1996-01-01")

    nations_pdf = df_nation.loc[
        df_nation["n_name"].isin([nation_a, nation_b]), ["n_nationkey", "n_name"]
    ].to_pandas()

    nation_key_by_name: dict[str, int] = {
        str(row["n_name"]): int(row["n_nationkey"]) for _, row in nations_pdf.iterrows()
    }
    if nation_a not in nation_key_by_name or nation_b not in nation_key_by_name:
        return _empty_result()

    nation_a_key = nation_key_by_name[nation_a]
    nation_b_key = nation_key_by_name[nation_b]

    sup_base = df_supplier[["s_suppkey", "s_nationkey"]]
    sup_mask = (sup_base["s_nationkey"] == nation_a_key) | (sup_base["s_nationkey"] == nation_b_key)
    sup = sup_base.loc[sup_mask]
    sup = sup.assign(supp_code=(sup["s_nationkey"] == nation_b_key).astype("int32"))
    sup = sup[["s_suppkey", "supp_code"]].rename(columns={"s_suppkey": "l_suppkey"})

    cust_base = df_customer[["c_custkey", "c_nationkey"]]
    cust_mask = (cust_base["c_nationkey"] == nation_a_key) | (
        cust_base["c_nationkey"] == nation_b_key
    )
    cust = cust_base.loc[cust_mask]
    cust = cust.assign(cust_code=(cust["c_nationkey"] == nation_b_key).astype("int32"))
    cust = cust[["c_custkey", "cust_code"]].rename(columns={"c_custkey": "o_custkey"})

    if len(sup) == 0 or len(cust) == 0:
        return _empty_result()

    oc = df_orders[["o_orderkey", "o_custkey"]].merge(cust, on="o_custkey", how="inner")
    oc = oc[["o_orderkey", "cust_code"]].rename(columns={"o_orderkey": "l_orderkey"})

    ship_days_all = ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32"))
    date_mask = (ship_days_all >= shipdate_lo_days) & (ship_days_all <= shipdate_hi_days)

    li = df_lineitem.loc[
        date_mask, ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
    ]
    li = li.assign(l_shipdate_days=ship_days_all.loc[date_mask])

    if len(li) == 0 or len(oc) == 0:
        return _empty_result()

    li_sup = li.merge(sup, on="l_suppkey", how="inner")
    joined = li_sup.merge(oc, on="l_orderkey", how="inner")

    if len(joined) == 0:
        return _empty_result()

    supp_t = torch.as_tensor(joined["supp_code"].to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(joined["cust_code"].to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(joined["l_shipdate_days"].to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    revenue_t = _launch_q7_agg(
        supp_t,
        cust_t,
        ship_t,
        price_t,
        disc_t,
        shipdate_lo_days,
        shipdate_hi_days,
        year_split_days,
    )

    revenue_cpu = revenue_t.detach().cpu().numpy()

    supp_names: list[str] = []
    cust_names: list[str] = []
    years: list[int] = []
    revenues: list[float] = []

    for key in (2, 3, 4, 5):
        rev = float(revenue_cpu[key])
        if rev > 0.0:
            supp_code = key // 4
            cust_code = (key // 2) % 2
            year = 1995 + (key % 2)
            supp_names.append(nation_a if supp_code == 0 else nation_b)
            cust_names.append(nation_a if cust_code == 0 else nation_b)
            years.append(year)
            revenues.append(rev)

    if not revenues:
        return _empty_result()

    result = cudf.DataFrame(
        {
            "supp_nation": cudf.Series(supp_names),
            "cust_nation": cudf.Series(cust_names),
            "l_year": cudf.Series(years, dtype="int32"),
            "revenue": cudf.Series(revenues, dtype="float64"),
        }
    )
    result = result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
    return result
