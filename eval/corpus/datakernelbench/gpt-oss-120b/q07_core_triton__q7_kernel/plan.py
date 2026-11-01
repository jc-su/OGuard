import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton  # needed for triton.cdiv in launch grid calculation

# -------------------------------------------------
# Triton kernel source (written to disk at import)
# -------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _q7_kernel(
    supp_ptr,               # int32 *
    cust_ptr,               # int32 *
    shipdate_ptr,           # int32 *
    price_ptr,              # float64 *
    disc_ptr,               # float64 *
    date_lo,                # int32   (scalar)
    date_hi,                # int32   (scalar)
    year_boundary,          # int32   (scalar, first day of 1996)
    out_ptr,                # float64 * (size 8)
    n_elements,             # int64   (scalar)
    BLOCK_SIZE: tl.constexpr = 8192,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load shipdate and apply date filter
    ship = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    keep = mask & (ship >= date_lo) & (ship <= date_hi)

    # conditional loads for rows that satisfy the filter
    supp = tl.load(supp_ptr + offs, mask=keep, other=0)
    cust = tl.load(cust_ptr + offs, mask=keep, other=0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)

    # compute revenue contribution
    vol = price * (1.0 - disc)

    # determine year: 1995 or 1996
    year = tl.where(ship >= year_boundary, 1996, 1995)
    year_offset = year - 1995          # 0 or 1

    # compact group key : supp*4 + cust*2 + year_offset  (0..7)
    key = supp * 4 + cust * 2 + year_offset

    # scatter‑add into per‑group accumulator
    tl.atomic_add(out_ptr + key, vol)
'''

def _load_kernels():
    """Write the Triton kernel source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="q7_triton_")
    path = os.path.join(tmpdir, "_q7_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q7_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels once at module import
_K = _load_kernels()

def _launch_q7(
    supp_nation_code: torch.Tensor,
    cust_nation_code: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    year_boundary_days: int,
) -> torch.Tensor:
    """
    Fuse filter, projection, year extraction, group‑by and aggregation
    into a single Triton kernel. Returns an 8‑element tensor where each
    entry corresponds to a possible (supp, cust, year) combination.
    """
    n = supp_nation_code.numel()
    out = torch.zeros(8, dtype=torch.float64, device=supp_nation_code.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._q7_kernel[grid](
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        shipdate_lo_days,
        shipdate_hi_days,
        year_boundary_days,
        out,
        n,
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
    Core computation for TPC‑H Q7.
    A single fused Triton kernel performs:
      * date filtering,
      * year derivation (1995/1996),
      * compact group key construction,
      * scatter‑add into an 8‑element revenue vector.
    """
    # first day of 1996 as epoch days
    year_boundary_days = int(pd.to_datetime('1996-01-01').timestamp() // 86400)

    revenue = _launch_q7(
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        shipdate_lo_days,
        shipdate_hi_days,
        year_boundary_days,
    )

    # decode the 8 possible groups
    gid = torch.arange(8, dtype=torch.int64, device=revenue.device)
    out_supp = (gid // 4).to(torch.int32)                 # 0 or 1
    out_cust = ((gid // 2) % 2).to(torch.int32)           # 0 or 1
    out_year = (gid % 2).to(torch.int32) + 1995          # 1995 or 1996

    # keep only valid nation‑pair combos and groups with data
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

    # keep only the two nations of interest
    nations2 = df_nation[df_nation["n_name"].isin([nation_a, nation_b])][
        ["n_nationkey", "n_name"]
    ]

    # supplier ↔ nation (supp_nation)
    sup = df_supplier[["s_suppkey", "s_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "s_nationkey", "n_name": "supp_nation"}),
        on="s_nationkey",
    )

    # customer ↔ nation (cust_nation)
    cust = df_customer[["c_custkey", "c_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "c_nationkey", "n_name": "cust_nation"}),
        on="c_nationkey",
    )

    # shipdate bounds as epoch days
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # lineitem + supplier nation
    li = df_lineitem[["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]]
    li = li.merge(sup[["s_suppkey", "supp_nation"]], left_on="l_suppkey", right_on="s_suppkey")

    # orders + customer nation
    oc = df_orders[["o_orderkey", "o_custkey"]].merge(
        cust[["c_custkey", "cust_nation"]], left_on="o_custkey", right_on="c_custkey"
    )

    # final join
    joined = li.merge(oc[["o_orderkey", "cust_nation"]], left_on="l_orderkey", right_on="o_orderkey")

    # filter to the two cross‑nation pairs
    pair_mask = (
        ((joined["supp_nation"] == nation_a) & (joined["cust_nation"] == nation_b))
        | ((joined["supp_nation"] == nation_b) & (joined["cust_nation"] == nation_a))
    )
    joined = joined[pair_mask]

    # encode nations: nation_a -> 0, nation_b -> 1
    supp_code = (joined["supp_nation"] == nation_b).astype("int32")
    cust_code = (joined["cust_nation"] == nation_b).astype("int32")

    # shipdate → days as int32
    l_shipdate_days = ((joined["l_shipdate"].astype("int64") // 86400).astype("int32"))

    # move columns to GPU tensors
    supp_t = torch.as_tensor(supp_code.to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(cust_code.to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    # core fused computation
    supp_out, cust_out, year_out, rev_out = _query_core(
        supp_t, cust_t, ship_t, price_t, disc_t, shipdate_lo_days, shipdate_hi_days
    )

    # map integer codes back to nation names
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

    return result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
