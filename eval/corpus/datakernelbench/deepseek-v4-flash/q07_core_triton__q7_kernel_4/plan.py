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

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["ker_n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q7_kernel(
    supp_ptr, cust_ptr, shipdate_ptr, price_ptr, disc_ptr,
    date_lo, date_hi, out_ptr, ker_n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < ker_n_elements
    
    # Load filter columns
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    supp = tl.load(supp_ptr + offs, mask=mask, other=0)
    cust = tl.load(cust_ptr + offs, mask=mask, other=0)
    
    # Filter: date range and valid nation pairs
    ok_date = (shipdate >= date_lo) & (shipdate <= date_hi)
    ok_pair = ((supp == 0) & (cust == 1)) | ((supp == 1) & (cust == 0))
    keep = mask & ok_date & ok_pair
    
    # Conditional load of price and discount
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    volume = price * (1.0 - disc)
    
    # Compute year from shipdate days using Howard Hinnant algorithm
    # Use tl.cast to convert and tl.where for conditional
    ship_int64 = shipdate.to(tl.int64)
    days = tl.where(keep, ship_int64, tl.zeros_like(ship_int64))
    days = days + 719468
    era = tl.where(days >= 0, days, days - 146096) // 146097
    doe = (days - era * 146097).to(tl.int64)
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(m <= 2, y + 1, y).to(tl.int32)
    
    # Compute key: supp*4 + cust*2 + (year - 1995)
    year_offset = year - 1995
    key = supp.to(tl.int64) * 4 + cust.to(tl.int64) * 2 + year_offset.to(tl.int64)
    
    # Scatter-add into output array (8 groups)
    vol_fp64 = volume.to(tl.float64)
    # Drop rows that don't contribute (key >= 8)
    tl.atomic_add(out_ptr + key, vol_fp64, mask=keep & (key < 8))
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q7_triton_")
    path = os.path.join(tmpdir, "_q7_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q7_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _launch_q7(
    supp: torch.Tensor,
    cust: torch.Tensor,
    shipdate_days: torch.Tensor,
    price: torch.Tensor,
    disc: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    n = price.numel()
    out = torch.zeros(8, dtype=torch.float64, device=price.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q7_kernel[grid](
        supp, cust, shipdate_days, price, disc,
        shipdate_lo_days, shipdate_hi_days, out, n,
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
    Fused Triton kernel for TPC-H Q7 hot path.
    Runs entirely on GPU: filters, projects (volume, year), and aggregates into 8 groups.
    Returns filtered group vectors (supp, cust, year, revenue) matching reference semantics.
    """
    revenue = _launch_q7(
        supp_nation_code, cust_nation_code, l_shipdate_days,
        l_extendedprice, l_discount, shipdate_lo_days, shipdate_hi_days,
    )
    
    # Build output arrays from the 8 groups
    device = l_extendedprice.device
    group_ids = torch.arange(8, dtype=torch.int64, device=device)
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
