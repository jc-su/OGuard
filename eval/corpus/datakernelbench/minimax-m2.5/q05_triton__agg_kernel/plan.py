import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source – must be written to disk before import.
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
def _agg_kernel(
    nation_id_ptr,
    price_ptr,
    discount_ptr,
    out_ptr,
    n_elements,
    num_nations,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused filter → project → aggregate kernel for Q5.
    
    Each thread loads a row, computes revenue = price * (1 - discount),
    and atomically adds to the per‑nation accumulator.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load row data (masked loads ensure OOB lanes get neutral values)
    nation = tl.load(nation_id_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Project: compute revenue for kept rows
    revenue = price * (1.0 - discount)

    # Global per‑nation aggregation via atomic add
    tl.atomic_add(out_ptr + nation, revenue)
'''

# ----------------------------------------------------------------------
# Utility to load the kernels from the on‑disk source.
# ----------------------------------------------------------------------
def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q5_triton_")
    path = os.path.join(tmpdir, "_q5_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q5_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

# ----------------------------------------------------------------------
# Triton launcher for the fused aggregation.
# ----------------------------------------------------------------------
def _agg(
    nation_ids: torch.Tensor,
    price: torch.Tensor,
    discount: torch.Tensor,
    num_nations: int,
) -> torch.Tensor:
    """Perform per‑nation revenue aggregation on the GPU using Triton."""
    n = nation_ids.numel()
    # Output buffer: one float64 accumulator per nation
    out = torch.zeros(num_nations, dtype=torch.float64, device=nation_ids.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._agg_kernel[grid](
        nation_ids,
        price,
        discount,
        out,
        n,
        num_nations,
    )
    return out

# ----------------------------------------------------------------------
# Query implementation (TPC‑H Q5).
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "ASIA",
    orderdate_lo: str = "1994-01-01",
    orderdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT n_name,
               sum(l_extendedprice * (1 - l_discount)) AS revenue
        FROM customer, orders, lineitem, supplier, nation, region
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND l_suppkey = s_suppkey
          AND c_nationkey = s_nationkey
          AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey
          AND r_name = 'ASIA'
          AND o_orderdate >= DATE '1994-01-01'
          AND o_orderdate <  DATE '1995-01-01'
        GROUP BY n_name
        ORDER BY revenue DESC;

    Description:
        TPC‑H Q5 – Local supplier volume for a region over one year.
        This implementation uses cuDF for joins and filtering, then a
        fused Triton kernel for the final projection‑aggregation step,
        which reduces memory traffic by materialising only the output
        accumulator array.

    Params:
        tables (dict[str, cudf.DataFrame]): Dictionary of input tables.
        region_name (str): Region to filter on (default 'ASIA').
        orderdate_lo (str): Inclusive lower bound on order date (default '1994-01-01').
        orderdate_hi (str): Exclusive upper bound on order date (default '1995-01-01').
    """
    # ------------------------------------------------------------------
    # 1. Filter region → nation.
    # ------------------------------------------------------------------
    df_region = tables["region"]
    region_f = df_region[df_region["r_name"] == region_name]

    df_nation = tables["nation"]
    nation_f = df_nation.merge(
        region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey"
    )

    # ------------------------------------------------------------------
    # 2. Filter orders by date range (int32 epoch days).
    # ------------------------------------------------------------------
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    df_orders = tables["orders"]
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (
        orders_f["o_orderdate"].astype("int64") // 86400
    ).astype("int32")
    orders_f = orders_f[
        (orders_f["o_orderdate_days"] >= lo_days)
        & (orders_f["o_orderdate_days"] < hi_days)
    ]

    # ------------------------------------------------------------------
    # 3. Join chain: customer → orders → lineitem → supplier → nation.
    # ------------------------------------------------------------------
    df_customer = tables["customer"]
    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f[["o_orderkey", "o_custkey"]],
        left_on="c_custkey",
        right_on="o_custkey",
    )

    df_lineitem = tables["lineitem"]
    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey",
        right_on="l_orderkey",
    )

    df_supplier = tables["supplier"]
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )
    # Keep only rows where customer nation matches supplier nation
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    # ------------------------------------------------------------------
    # 4. Attach nation name.
    # ------------------------------------------------------------------
    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    # ------------------------------------------------------------------
    # 5. Build compact nation identifiers (0‑based) for the kernel.
    # ------------------------------------------------------------------
    nations_unique = (
        nation_f[["n_nationkey", "n_name"]]
        .drop_duplicates()
        .sort_values("n_nationkey")
        .reset_index(drop=True)
    )
    nations_unique["nation_id"] = cudf.Series(
        range(len(nations_unique)), dtype="int64"
    )

    final_df = final_df.merge(
        nations_unique[["n_nationkey", "nation_id"]], on="n_nationkey"
    )
    num_nations = len(nations_unique)

    # ------------------------------------------------------------------
    # 6. GPU aggregation using fused Triton kernel.
    # ------------------------------------------------------------------
    nation_ids_t = torch.as_tensor(
        final_df["nation_id"].to_cupy(), dtype=torch.int64
    )
    price_t = torch.as_tensor(
        final_df["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    discount_t = torch.as_tensor(
        final_df["l_discount"].to_cupy(), dtype=torch.float64
    )

    revenue_per_nation = _agg(
        nation_ids_t, price_t, discount_t, num_nations
    )

    # ------------------------------------------------------------------
    # 7. Sort by revenue descending and map back to nation names.
    # ------------------------------------------------------------------
    order = torch.argsort(revenue_per_nation, descending=True)
    sorted_nation_ids = torch.arange(
        num_nations, dtype=torch.int64, device=revenue_per_nation.device
    )[order]
    sorted_revenue = revenue_per_nation[order]

    result_ids = cudf.Series(sorted_nation_ids, dtype="int64")
    result_rev = cudf.Series(sorted_revenue, dtype="float64")
    result = cudf.DataFrame({"nation_id": result_ids, "revenue": result_rev})
    result = result.merge(
        nations_unique[["nation_id", "n_name"]], on="nation_id"
    )
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame({
        "n_name": result["n_name"],
        "revenue": result["revenue"],
    })
