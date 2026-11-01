import os
import tempfile
import importlib.util
import linecache

import torch
import triton
import cudf
import pandas as pd

# ---------------------------------------------------------------------------
# Triton kernel source (stored on disk at import time)
# ---------------------------------------------------------------------------
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
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q7_kernel(
    supp_ptr, cust_ptr, shipdate_ptr,
    price_ptr, discount_ptr,
    date_lo, date_hi,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load filtering columns
    supp = tl.load(supp_ptr + offs, mask=mask, other=0)
    cust = tl.load(cust_ptr + offs, mask=mask, other=0)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)

    # Filter by shipdate range
    keep = (shipdate >= date_lo) & (shipdate <= date_hi)

    # ----- year extraction (civil algorithm) -----
    days = shipdate.to(tl.int64) + 719468
    era = tl.where(days >= 0, days, days - 146096) // 146097
    doe = days - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(m <= 2, y + 1, y)
    year_offset = year - 1995

    # Compute group key
    key = tl.where(keep, supp * 4 + cust * 2 + year_offset, 0)

    # Load price & discount only where needed
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    volume = price * (1.0 - discount)

    # ----- per-group reduction within the block -----
    # For each of the 8 possible groups, compute the sum for this block
    for i in tl.static_range(8):
        group_mask = (key == i)
        group_sum = tl.sum(tl.where(group_mask, volume, 0.0))
        # Only atomic add if there's something to add
        if group_sum != 0:
            tl.atomic_add(out_ptr + i, group_sum)
'''

def _load_kernels():
    """Write the kernel source to a temporary file and import it."""
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

# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------
def _launch_q7(
    supp: torch.Tensor,
    cust: torch.Tensor,
    shipdate: torch.Tensor,
    price: torch.Tensor,
    discount: torch.Tensor,
    date_lo: int,
    date_hi: int,
) -> torch.Tensor:
    """Launch the fused filter-project-aggregate Triton kernel."""
    n = supp.numel()
    out = torch.zeros(8, dtype=torch.float64, device=supp.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q7_kernel[grid](
        supp, cust, shipdate, price, discount,
        date_lo, date_hi, out, n,
    )
    return out


# ---------------------------------------------------------------------------
# Core logic (filter + aggregate) – replaces reference _query_core
# ---------------------------------------------------------------------------
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
    Fusion of filter, projection (volume) and group-by aggregation.
    Returns four tensors: supp_code, cust_code, year, revenue.
    """
    # Run the fused Triton kernel
    revenue = _launch_q7(
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # Reconstruct the group identifiers (same scheme as reference)
    group_ids = torch.arange(8, dtype=torch.int64, device=revenue.device)
    out_supp = (group_ids // 4).to(torch.int32)          # 0 or 1
    out_cust = ((group_ids // 2) % 2).to(torch.int32)    # 0 or 1
    out_year = (group_ids % 2).to(torch.int32) + 1995    # 1995 or 1996

    # Keep only valid nation pairs (FR<->DE) that have non-zero revenue
    valid = ((out_supp == 0) & (out_cust == 1)) | ((out_supp == 1) & (out_cust == 0))
    valid = valid & (revenue > 0)

    return out_supp[valid], out_cust[valid], out_year[valid], revenue[valid]


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------
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
            SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
                   extract(year FROM l_shipdate) AS l_year,
                   l_extendedprice * (1 - l_discount) AS volume
            FROM supplier, lineitem, orders, customer, nation n1, nation n2
            WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey
                  AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey
                  AND c_nationkey = n2.n_nationkey
                  AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
                       OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
                  AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
        ) GROUP BY supp_nation, cust_nation, l_year
        ORDER BY supp_nation, cust_nation, l_year;

    Description:
        Computes the shipping revenue between two nations (FRANCE and GERMANY)
        for each year in the given date range.  The hot path (filter -> project
        -> group-by sum) is executed as a single fused Triton kernel, reducing
        memory traffic and avoiding multiple passes over the data.

    Params:
        tables: dict of cuDF DataFrames (customer, lineitem, nation, orders, supplier).
        nation_a: First nation name (default 'FRANCE').
        nation_b: Second nation name (default 'GERMANY').
        shipdate_lo: Inclusive lower bound for l_shipdate (default '1995-01-01').
        shipdate_hi: Inclusive upper bound for l_shipdate (default '1996-12-31').
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]

    # ----- cuDF joins (same as reference) -----
    nations2 = df_nation[df_nation["n_name"].isin([nation_a, nation_b])][
        ["n_nationkey", "n_name"]
    ]

    sup = df_supplier[["s_suppkey", "s_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "s_nationkey", "n_name": "supp_nation"}),
        on="s_nationkey",
    )

    cust = df_customer[["c_custkey", "c_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "c_nationkey", "n_name": "cust_nation"}),
        on="c_nationkey",
    )

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    li = df_lineitem[["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]]
    li = li.merge(sup[["s_suppkey", "supp_nation"]], left_on="l_suppkey", right_on="s_suppkey")

    oc = df_orders[["o_orderkey", "o_custkey"]].merge(
        cust[["c_custkey", "cust_nation"]], left_on="o_custkey", right_on="c_custkey"
    )

    joined = li.merge(oc[["o_orderkey", "cust_nation"]], left_on="l_orderkey", right_on="o_orderkey")

    # Keep only the two directional country pairs
    pair_mask = (
        ((joined["supp_nation"] == nation_a) & (joined["cust_nation"] == nation_b))
        | ((joined["supp_nation"] == nation_b) & (joined["cust_nation"] == nation_a))
    )
    joined = joined[pair_mask]

    # Encode nation codes: 0 = nation_a, 1 = nation_b
    supp_code = (joined["supp_nation"] == nation_b).astype("int32")
    cust_code = (joined["cust_nation"] == nation_b).astype("int32")
    l_shipdate_days = ((joined["l_shipdate"].astype("int64") // 86400).astype("int32"))

    # ----- move data to GPU -----
    supp_t = torch.as_tensor(supp_code.to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(cust_code.to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    # ----- fused Triton kernel for filter + project + aggregate -----
    supp_out, cust_out, year_out, rev_out = _query_core(
        supp_t, cust_t, ship_t, price_t, disc_t,
        shipdate_lo_days, shipdate_hi_days,
    )

    # ----- map codes back to nation names -----
    supp_np = supp_out.cpu().numpy()
    cust_np = cust_out.cpu().numpy()
    supp_names = [nation_a if v == 0 else nation_b for v in supp_np]
    cust_names = [nation_a if v == 0 else nation_b for v in cust_np]

    # ----- build result DataFrame -----
    result = cudf.DataFrame({
        "supp_nation": cudf.Series(supp_names),
        "cust_nation": cudf.Series(cust_names),
        "l_year": year_out,
        "revenue": rev_out,
    })

    result = result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
    return result
