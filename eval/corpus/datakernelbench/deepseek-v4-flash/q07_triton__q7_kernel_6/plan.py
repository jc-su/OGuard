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
    key=["n_elements"],
    reset_to_zero=["out_revenue_ptr"],
)
@triton.jit
def _q7_kernel(
    supp_code_ptr, cust_code_ptr, shipdate_days_ptr, price_ptr, disc_ptr,
    n_elements,
    shipdate_lo, shipdate_hi,
    out_revenue_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    supp = tl.load(supp_code_ptr + offs, mask=mask, other=2)
    cust = tl.load(cust_code_ptr + offs, mask=mask, other=2)
    ship = tl.load(shipdate_days_ptr + offs, mask=mask, other=shipdate_hi + 1)

    # filter: shipdate range and (supp=0,cust=1) or (supp=1,cust=0)
    date_ok = (ship >= shipdate_lo) & (ship <= shipdate_hi)
    pair_ok = ((supp == 0) & (cust == 1)) | ((supp == 1) & (cust == 0))
    keep = mask & date_ok & pair_ok

    # early exit if nothing to load
    if tl.sum(keep.to(tl.int64)) == 0:
        return

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    supp_sel = tl.load(supp_code_ptr + offs, mask=keep, other=0)
    cust_sel = tl.load(cust_code_ptr + offs, mask=keep, other=0)

    # compute year from shipdate days
    ship64 = ship.to(tl.int64)
    ship64 = ship64 + 719468
    era = tl.where(ship64 >= 0, ship64, ship64 - 146096) // 146097
    doe = ship64 - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(m <= 2, y + 1, y)
    year_offset = year - 1995

    volume = price * (1.0 - disc)

    # group key: supp*4 + cust*2 + year_offset
    key = supp_sel.to(tl.int64) * 4 + cust_sel.to(tl.int64) * 2 + year_offset.to(tl.int64)

    # per-warp block reduction into local group accumulators
    num_groups = 8
    # Use local accumulators for each group
    # We can't use tl.zeros with non-constexpr shape, use a loop over groups
    for g in tl.static_range(8):
        g_mask = keep & (key == g)
        g_sum = tl.sum(tl.where(g_mask, volume, 0.0))
        if g_sum != 0.0:
            tl.atomic_add(out_revenue_ptr + g, g_sum)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q7_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _launch_q7(
    supp_code: torch.Tensor,
    cust_code: torch.Tensor,
    shipdate_days: torch.Tensor,
    price: torch.Tensor,
    disc: torch.Tensor,
    shipdate_lo: int,
    shipdate_hi: int,
) -> torch.Tensor:
    n = supp_code.numel()
    out_revenue = torch.zeros(8, dtype=torch.float64, device=supp_code.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q7_kernel[grid](
        supp_code, cust_code, shipdate_days, price, disc,
        n,
        shipdate_lo, shipdate_hi,
        out_revenue,
    )
    return out_revenue

def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_a: str = "FRANCE",
    nation_b: str = "GERMANY",
    shipdate_lo: str = "1995-01-01",
    shipdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL: same as reference (TPC-H Q7).
    Description: Joins in cuDF (strings, small dims), then fused Triton kernel for filter/project/aggregate.
    Fused kernel does: load 4 columns, filter by date + nation pair, compute year, apply projection, atomic_add to 8 groups.
    Params: see reference.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]

    nations2 = df_nation[df_nation["n_name"].isin([nation_a, nation_b])][["n_nationkey", "n_name"]]

    sup = df_supplier[["s_suppkey", "s_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "s_nationkey", "n_name": "supp_nation"}),
        on="s_nationkey",
    )

    cust = df_customer[["c_custkey", "c_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "c_nationkey", "n_name": "cust_nation"}),
        on="c_nationkey",
    )

    li = df_lineitem[["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]]
    li = li.merge(sup[["s_suppkey", "supp_nation"]], left_on="l_suppkey", right_on="s_suppkey")

    oc = df_orders[["o_orderkey", "o_custkey"]].merge(
        cust[["c_custkey", "cust_nation"]], left_on="o_custkey", right_on="c_custkey"
    )

    joined = li.merge(oc[["o_orderkey", "cust_nation"]], left_on="l_orderkey", right_on="o_orderkey")

    pair_mask = (
        ((joined["supp_nation"] == nation_a) & (joined["cust_nation"] == nation_b))
        | ((joined["supp_nation"] == nation_b) & (joined["cust_nation"] == nation_a))
    )
    joined = joined[pair_mask]

    supp_code = (joined["supp_nation"] == nation_b).astype("int32")
    cust_code = (joined["cust_nation"] == nation_b).astype("int32")

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_shipdate_days = ((joined["l_shipdate"].astype("int64") // 86400).astype("int32"))

    supp_t = torch.as_tensor(supp_code.to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(cust_code.to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    out_revenue = _launch_q7(supp_t, cust_t, ship_t, price_t, disc_t,
                             shipdate_lo_days, shipdate_hi_days)

    group_ids = torch.arange(8, dtype=torch.int64, device=out_revenue.device)
    out_supp = (group_ids // 4).to(torch.int32)
    out_cust = ((group_ids // 2) % 2).to(torch.int32)
    out_year = (group_ids % 2).to(torch.int32) + 1995

    valid = ((out_supp == 0) & (out_cust == 1)) | ((out_supp == 1) & (out_cust == 0))
    valid = valid & (out_revenue > 0)

    supp_out = out_supp[valid]
    cust_out = out_cust[valid]
    year_out = out_year[valid]
    rev_out = out_revenue[valid]

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
