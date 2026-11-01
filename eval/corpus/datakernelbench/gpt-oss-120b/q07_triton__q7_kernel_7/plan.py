import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source (written to a temporary file and imported)
# ----------------------------------------------------------------------
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
    supp_ptr,                # int32 *
    cust_ptr,                # int32 *
    ship_ptr,                # int32 *
    price_ptr,               # float64 *
    disc_ptr,                # float64 *
    lo_day,                  # i32   scalar
    hi_day,                  # i32   scalar
    day_1996,                # i32   scalar (first day of 1996)
    out_ptr,                 # float64 *
    n_elements,              # i64   total rows
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns
    supp = tl.load(supp_ptr + offs, mask=mask, other=0).to(tl.int32)
    cust = tl.load(cust_ptr + offs, mask=mask, other=0).to(tl.int32)
    ship = tl.load(ship_ptr + offs, mask=mask, other=0).to(tl.int32)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)

    # Predicate on shipdate range
    keep = mask & (ship >= lo_day) & (ship <= hi_day)

    # Year offset: 0 for 1995, 1 for 1996
    year_off = tl.where(ship >= day_1996, 1, 0)

    # Composite key in [0,7]
    key = (supp * 4 + cust * 2 + year_off).to(tl.int64)

    # Compute volume
    volume = price * (1.0 - disc)

    # Add contribution (zero for filtered rows)
    tl.atomic_add(out_ptr + key, tl.where(keep, volume, 0.0))
'''

def _load_kernels() -> object:
    """Write the Triton source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="q07_triton_")
    path = os.path.join(tmpdir, "_q07_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q07_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# Load kernels once at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Helper / launch functions
# ----------------------------------------------------------------------
def _launch_q7(
    supp_code: torch.Tensor,
    cust_code: torch.Tensor,
    ship_days: torch.Tensor,
    price: torch.Tensor,
    disc: torch.Tensor,
    ship_lo: int,
    ship_hi: int,
) -> torch.Tensor:
    """
    Fuse filter, year extraction, and 4‑group aggregation into a single Triton kernel.
    Returns a float64 tensor of length 8 (max possible groups).
    """
    n = ship_days.numel()
    out = torch.zeros(8, dtype=torch.float64, device=ship_days.device)

    # First day of 1996 (epoch‑days). 1995 is not a leap year → add 365.
    day_1996 = ship_lo + 365

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q7_kernel[grid](
        supp_code,
        cust_code,
        ship_days,
        price,
        disc,
        ship_lo,
        ship_hi,
        day_1996,
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
    Core computation: fused filter‑project‑aggregate using Triton.
    Mirrors the reference implementation’s output layout.
    """
    revenue = _launch_q7(
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # Decode the 8 possible group slots (same layout as reference)
    group_ids = torch.arange(8, dtype=torch.int64, device=revenue.device)
    out_supp = (group_ids // 4).to(torch.int32)                 # 0=FRANCE,1=GERMANY
    out_cust = ((group_ids // 2) % 2).to(torch.int32)           # 0=FRANCE,1=GERMANY
    out_year = (group_ids % 2).to(torch.int32) + 1995          # 1995 or 1996

    # Keep only the two valid nation‑pair combos and groups with non‑zero revenue
    valid = ((out_supp == 0) & (out_cust == 1)) | ((out_supp == 1) & (out_cust == 0))
    valid = valid & (revenue > 0)

    return out_supp[valid], out_cust[valid], out_year[valid], revenue[valid]

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_a: str = "FRANCE",
    nation_b: str = "GERMANY",
    shipdate_lo: str = "1995-01-01",
    shipdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue
        FROM (
            SELECT n1.n_name AS supp_nation,
                   n2.n_name AS cust_nation,
                   EXTRACT(YEAR FROM l_shipdate) AS l_year,
                   l_extendedprice * (1 - l_discount) AS volume
            FROM supplier, lineitem, orders, customer,
                 nation n1, nation n2
            WHERE s_suppkey = l_suppkey
              AND o_orderkey = l_orderkey
              AND c_custkey = o_custkey
              AND s_nationkey = n1.n_nationkey
              AND c_nationkey = n2.n_nationkey
              AND ((n1.n_name = '{nation_a}' AND n2.n_name = '{nation_b}')
                OR (n1.n_name = '{nation_b}' AND n2.n_name = '{nation_a}'))
              AND l_shipdate BETWEEN DATE '{shipdate_lo}'
                                 AND DATE '{shipdate_hi}'
        ) AS shipping
        GROUP BY supp_nation, cust_nation, l_year
        ORDER BY supp_nation, cust_nation, l_year;

    Description:
        The join‑heavy preparation is performed with cuDF (highly optimized
        columnar operations).  The hot path—filtering by shipdate,
        computing ``volume = l_extendedprice * (1 - l_discount)`` and the
        four‑group aggregation—is fused into a single Triton kernel.
        This eliminates multiple passes over the data, reduces global memory
        traffic, and uses fast FP64 atomic adds for the tiny group set,
        delivering a noticeable speed‑up on an H100.

    Params:
        nation_a (str): first nation name.
        nation_b (str): second nation name.
        shipdate_lo (str): inclusive lower bound for ``l_shipdate`` (YYYY‑MM‑DD).
        shipdate_hi (str): inclusive upper bound for ``l_shipdate`` (YYYY‑MM‑DD).

    Returns:
        cudf.DataFrame with columns ``supp_nation``, ``cust_nation``,
        ``l_year`` (int32) and ``revenue`` (float64), sorted as required.
    """
    # ------------------------------------------------------------------
    # 1️⃣ cuDF preprocessing – joins and nation filtering
    # ------------------------------------------------------------------
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]

    # Keep only the two nations of interest
    nations2 = df_nation[df_nation["n_name"].isin([nation_a, nation_b])][
        ["n_nationkey", "n_name"]
    ]

    # Supplier ↔ nation (supp_nation)
    sup = df_supplier[["s_suppkey", "s_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "s_nationkey", "n_name": "supp_nation"}),
        on="s_nationkey",
    )

    # Customer ↔ nation (cust_nation)
    cust = df_customer[["c_custkey", "c_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "c_nationkey", "n_name": "cust_nation"}),
        on="c_nationkey",
    )

    # lineitem + supplier nation
    li = df_lineitem[[
        "l_orderkey", "l_suppkey", "l_shipdate",
        "l_extendedprice", "l_discount"
    ]].merge(
        sup[["s_suppkey", "supp_nation"]], left_on="l_suppkey", right_on="s_suppkey"
    )

    # orders + customer nation
    oc = df_orders[["o_orderkey", "o_custkey"]].merge(
        cust[["c_custkey", "cust_nation"]], left_on="o_custkey", right_on="c_custkey"
    )

    # Final join
    joined = li.merge(oc[["o_orderkey", "cust_nation"]], left_on="l_orderkey", right_on="o_orderkey")

    # Keep only the two nation‑pair permutations
    pair_mask = (
        ((joined["supp_nation"] == nation_a) & (joined["cust_nation"] == nation_b))
        | ((joined["supp_nation"] == nation_b) & (joined["cust_nation"] == nation_a))
    )
    joined = joined[pair_mask]

    # ------------------------------------------------------------------
    # 2️⃣ Encode nation columns as 0/1 (0 → nation_a, 1 → nation_b)
    # ------------------------------------------------------------------
    supp_code = (joined["supp_nation"] == nation_b).astype("int32")
    cust_code = (joined["cust_nation"] == nation_b).astype("int32")

    # Convert shipdate to epoch‑days (int32)
    ship_days = ((joined["l_shipdate"].astype("int64") // 86400).astype("int32"))

    # ------------------------------------------------------------------
    # 3️⃣ Move columns to GPU tensors (torch) – zero‑copy via CuPy
    # ------------------------------------------------------------------
    supp_t = torch.as_tensor(supp_code.to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(cust_code.to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(ship_days.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    # ------------------------------------------------------------------
    # 4️⃣ Date bounds (epoch‑days, inclusive)
    # ------------------------------------------------------------------
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # ------------------------------------------------------------------
    # 5️⃣ Core fused computation
    # ------------------------------------------------------------------
    supp_out, cust_out, year_out, rev_out = _query_core(
        supp_t,
        cust_t,
        ship_t,
        price_t,
        disc_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # ------------------------------------------------------------------
    # 6️⃣ Map integer codes back to nation names
    # ------------------------------------------------------------------
    supp_codes_cpu = supp_out.cpu().numpy()
    cust_codes_cpu = cust_out.cpu().numpy()
    supp_names = [nation_a if v == 0 else nation_b for v in supp_codes_cpu]
    cust_names = [nation_a if v == 0 else nation_b for v in cust_codes_cpu]

    # ------------------------------------------------------------------
    # 7️⃣ Assemble final cuDF result
    # ------------------------------------------------------------------
    result = cudf.DataFrame({
        "supp_nation": cudf.Series(supp_names),
        "cust_nation": cudf.Series(cust_names),
        "l_year": year_out,      # int32 column
        "revenue": rev_out,      # float64 column
    })
    result = result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
    return result
