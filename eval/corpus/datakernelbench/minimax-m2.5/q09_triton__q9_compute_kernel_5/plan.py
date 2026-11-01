import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
import numpy as np

# Triton kernel source: fused computation + groupby aggregation
# The kernel computes amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
# and aggregates by (nation_code, year) using atomic adds
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8, 16]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q9_fused_kernel(
    price_ptr, discount_ptr, supplycost_ptr, quantity_ptr,
    nation_code_ptr, year_ptr,
    num_groups, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance handles a block of rows
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns (using mask for safe out-of-bounds)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    nation_code = tl.load(nation_code_ptr + offs, mask=mask, other=-1)
    year = tl.load(year_ptr + offs, mask=mask, other=0)

    # Compute amount: l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
    amount = price * (1.0 - discount) - supplycost * quantity

    # Compute group index: nation_code * 100 + year (assuming year fits in 2 digits)
    # More robust: nation_code shifted left and OR with year
    # Using composite key: nation_code * 100 + year (covers years 0-99, nations 0-24)
    group_idx = nation_code * 100 + year

    # Aggregate using atomic add
    # Only valid rows contribute
    valid_mask = mask & (nation_code >= 0)
    
    # For each valid lane, atomic add to its group
    # We need to iterate since multiple lanes may have different groups
    for i in tl.static_range(BLOCK_SIZE):
        g = tl.load(group_idx + i * 0)  # This won't work, need different approach
        pass  # Placeholder

    # Simpler approach: use tl.where + reduction per program, then atomic
    # Since each program computes a partial sum per unique group in its block,
    # we need a two-pass or use a simpler atomic-per-element approach
    
    # For memory-bound scan with low selectivity, atomic per element is acceptable
    # Compute amount only for valid rows
    valid_amount = tl.where(valid_mask, amount, 0.0)
    
    # Atomic add to the global output at the group's offset
    # Need to compute which output slot
    # group_idx ranges from 0 to num_groups-1
    for i in tl.static_range(BLOCK_SIZE):
        g = tl.load(group_idx + i * 0)  # Can't index like this
        pass

# Simpler single-pass kernel: compute amounts and return for cuDF groupby
# This still fuses the arithmetic but lets cuDF handle the aggregation
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8, 16]
        for s in [2, 4]
    ],
    key=["n_elements"],
)
@triton.jit
def _q9_compute_kernel(
    price_ptr, discount_ptr, supplycost_ptr, quantity_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    # Compute amount inregisters: l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
    amount = price * (1.0 - discount) - supplycost * quantity

    # Store result
    tl.store(out_ptr + offs, amount, mask=mask)
'''

# Alternative kernel source with proper aggregation
_KERNEL_SOURCE_V2 = r'''
import triton
import triton.language as tl

# Kernel that computes amount and aggregates by (nation_code, year) in one pass
# Uses a fixed-size hash table in shared memory for local reduction,
# then atomically adds to global output

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs, "GROUPS": 64}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096]
        for w in [8, 16]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q9_fused_agg_kernel(
    price_ptr, discount_ptr, supplycost_ptr, quantity_ptr,
    nation_code_ptr, year_ptr,
    num_groups, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Shared memory for local aggregation
    local_agg = tl.zeros((GROUPS,), dtype=tl.float64)
    
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    nation_code = tl.load(nation_code_ptr + offs, mask=mask, other=-1)
    year = tl.load(year_ptr + offs, mask=mask, other=0)

    # Compute amount
    amount = price * (1.0 - discount) - supplycost * quantity

    # Compute group index
    group_idx = nation_code * 100 + year

    # Local reduction within the block
    for i in range(BLOCK_SIZE):
        g = group_idx[i]
        a = amount[i]
        valid = mask[i] & (g >= 0) & (g < GROUPS)
        if valid:
            local_agg[g] += a

    # Atomic add to global output
    for g in range(GROUPS):
        if local_agg[g] != 0.0:
            tl.atomic_add(out_ptr + g, local_agg[g])


# Simpler compute-only kernel that returns amounts as a column
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8, 16]
        for s in [2, 4]
    ],
    key=["n_elements"],
)
@triton.jit
def _q9_compute_kernel(
    price_ptr, discount_ptr, supplycost_ptr, quantity_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns - all float64
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    # Compute amount inregisters: l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
    # This is the hot path computation
    amount = price * (1.0 - discount) - supplycost * quantity

    # Store result
    tl.store(out_ptr + offs, amount, mask=mask)
'''


def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q9_triton_")
    path = os.path.join(tmpdir, "_q9_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE_V2)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q9_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q9_compute(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Launch Triton kernel to compute amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
    
    This fuses the multiply-subtract operation into a single kernel,
    avoiding separate kernel launches for each operation.
    """
    n = l_extendedprice.numel()
    out = torch.empty_like(l_extendedprice)  # Same dtype (float64)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    _K._q9_compute_kernel[grid](
        l_extendedprice, l_discount, ps_supplycost, l_quantity,
        out, n,
    )
    return out


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Core computation: compute amount for each row.
    
    Uses fused Triton kernel for the arithmetic: 
    amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
    
    This replaces the reference's plain Torch tensor operations with a 
    fused Triton kernel, reducing kernel launch overhead and enabling
    better memory coalescing.
    """
    return _launch_q9_compute(l_extendedprice, l_discount, ps_supplycost, l_quantity)


def run_query(
    tables: dict[str, cudf.DataFrame],
    name_pattern: str = "green"
) -> cudf.DataFrame:
    """
    SQL: SELECT nation, o_year, sum(amount) AS sum_profit FROM (SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM part, supplier, lineitem, partsupp, orders, nation WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey AND p_name LIKE '%green%') AS profit GROUP BY nation, o_year ORDER BY nation, o_year DESC;
    
    Description: TPC-H Q9 - Product Type Profit Measure. Computes profit by nation and year 
    for parts whose name matches a pattern.
    
    This optimized version:
    - Uses cuDF for joins and string filtering (efficient for string ops)
    - Uses fused Triton kernel for the amount computation (the hot path arithmetic)
    - The kernel fuses multiply-subtract in registers before storing to HBM
    
    Params: name_pattern (str): substring to filter part names (without % wildcards).
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # Filter parts by name pattern
    pat = name_pattern.replace("%", "")
    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]

    # Join chain: lineitem -> parts_filtered -> partsupp -> supplier -> orders -> nation
    li_parts = df_lineitem.merge(parts_filtered, left_on="l_partkey", right_on="p_partkey")

    li_ps = li_parts.merge(
        df_partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]],
        left_on=["l_partkey", "l_suppkey"],
        right_on=["ps_partkey", "ps_suppkey"],
    )

    li_ps_s = li_ps.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )

    li_ps_s_o = li_ps_s.merge(
        df_orders[["o_orderkey", "o_orderdate"]],
        left_on="l_orderkey",
        right_on="o_orderkey",
    )

    joined = li_ps_s_o.merge(
        df_nation[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    # Convert columns to tensors and compute using fused Triton kernel
    amount_t = _query_core(
        torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64),
    )

    # Build profit subquery frame in cuDF
    profit_df = cudf.DataFrame({
        "nation": joined["n_name"].reset_index(drop=True),
        "o_year": joined["o_orderdate"].dt.year.astype("int32").reset_index(drop=True),
        "amount": cudf.Series(amount_t),
    })

    # Group by nation, o_year and sum
    grouped = profit_df.groupby(["nation", "o_year"], as_index=False).agg({"amount": "sum"})
    grouped = grouped.rename(columns={"amount": "sum_profit"})

    # Order by nation ASC, o_year DESC
    grouped = grouped.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)

    return cudf.DataFrame({
        "nation": grouped["nation"],
        "o_year": grouped["o_year"],
        "sum_profit": grouped["sum_profit"],
    })
