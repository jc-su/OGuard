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
    reset_to_zero=["sum_brazil_ptr", "sum_total_ptr", "present_ptr"],
)
@triton.jit
def _q8_kernel(
    o_year_ptr,
    volume_ptr,
    is_brazil_ptr,
    sum_brazil_ptr,
    sum_total_ptr,
    present_ptr,
    min_year,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    year = tl.load(o_year_ptr + offs, mask=mask, other=0)
    vol = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=0)
    
    is_br = brazil > 0
    group_idx = year - min_year
    
    valid_group_idx = tl.where(mask, group_idx, 0)
    vol_br = tl.where(is_br, vol, 0.0)
    
    tl.atomic_add(sum_brazil_ptr + valid_group_idx, vol_br, mask=mask)
    tl.atomic_add(sum_total_ptr + valid_group_idx, vol, mask=mask)
    tl.atomic_max(present_ptr + valid_group_idx, 1, mask=mask)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q8_triton_")
    path = os.path.join(tmpdir, "_q8_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q8_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _query_core(
    o_year: torch.Tensor,
    volume: torch.Tensor,
    is_brazil: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU Triton kernel executor for the TPC-H Q8 hot path.
    
    What runs where:
    - Fused filtering and aggregation executes inside a custom Triton kernel purely on the GPU.
    - Python/cuDF handles string/date filtering and join logic before extracting tensors.
    
    Why it is faster:
    - Replaces multiple eager operations (torch.unique with return_inverse, torch.where, two scatter_adds)
      with a single, memory-bound pass.
    - Avoids instantiating numerous O(N) memory buffers.
    """
    n = o_year.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.int32, device=o_year.device), torch.empty(0, dtype=torch.float64, device=o_year.device)

    # Cast to int8 for safe Triton bool/int8 loading
    is_brazil_i8 = is_brazil.to(torch.int8)

    min_y = int(torch.min(o_year).item())
    max_y = int(torch.max(o_year).item())
    
    num_groups = max_y - min_y + 1
    
    # Pre-allocate output buffers
    sum_brazil = torch.zeros(num_groups, dtype=torch.float64, device=o_year.device)
    sum_total = torch.zeros(num_groups, dtype=torch.float64, device=o_year.device)
    present = torch.zeros(num_groups, dtype=torch.int32, device=o_year.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q8_kernel[grid](
        o_year, volume, is_brazil_i8,
        sum_brazil, sum_total, present,
        min_y, n
    )
    
    # Filter and construct return layout
    mask = present > 0
    years = torch.arange(min_y, max_y + 1, dtype=torch.int32, device=o_year.device)[mask]
    mkt_share = sum_brazil[mask] / sum_total[mask]
    
    return years, mkt_share


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
