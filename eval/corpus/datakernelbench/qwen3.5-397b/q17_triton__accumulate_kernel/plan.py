import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import triton.language as tl
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
    reset_to_zero=["sum_qty_ptr", "cnt_qty_ptr"],
)
@triton.jit
def _accumulate_kernel(
    partkey_ptr, quantity_ptr, is_filtered_ptr,
    sum_qty_ptr, cnt_qty_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    pk = tl.load(partkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    
    # Check if partkey is in filtered set
    filtered = tl.load(is_filtered_ptr + pk, mask=mask, other=0)
    
    # Only accumulate if partkey is filtered
    valid_mask = mask & (filtered != 0)
    
    tl.atomic_add(sum_qty_ptr + pk, qty, mask=valid_mask)
    tl.atomic_add(cnt_qty_ptr + pk, 1.0, mask=valid_mask)

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
def _filter_sum_kernel(
    partkey_ptr, quantity_ptr, price_ptr, is_filtered_ptr, thresholds_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    pk = tl.load(partkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    
    filtered = tl.load(is_filtered_ptr + pk, mask=mask, other=0)
    thresh = tl.load(thresholds_ptr + pk, mask=mask, other=0.0)
    
    # Condition: filtered AND qty < 0.2 * avg_qty (stored in thresh)
    cond = (filtered != 0) & (qty < thresh)
    
    contribution = tl.where(cond, price, 0.0)
    tl.atomic_add(out_ptr, tl.sum(contribution))
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q17_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                            FROM lineitem
                            WHERE l_partkey = p_partkey);
    Description: TPC-H Q17. Computes the small-quantity order revenue average for parts of a
        given brand and container. Uses fused Triton kernels for lineitem scans to reduce 
        memory bandwidth usage compared to multiple cuDF/PyTorch passes. Replaces searchsorted 
        with direct indexing on partkey domain (standard TPC-H property) for O(1) lookup.
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables 'lineitem' and 'part'.
        brand_val (str): Brand filter literal for p_brand.
        container_val (str): Container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Handle empty lineitem case
    if len(df_lineitem) == 0:
        return 0.0

    # Filter part table on host (string comparisons)
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    # If no parts match, result is 0
    if len(df_part_f) == 0:
        return 0.0

    # Get filtered partkeys
    filtered_pks_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    
    # Determine max partkey for direct indexing arrays
    # TPC-H partkeys are dense integers; use max(l_partkey) to size arrays safely
    max_pk = int(df_lineitem["l_partkey"].max())
    array_size = max_pk + 1

    # Create lookup array for filtered parts (1 if filtered, 0 otherwise)
    is_filtered = torch.zeros(array_size, dtype=torch.int32, device='cuda')
    is_filtered[filtered_pks_t] = 1

    # Load lineitem tensors
    l_partkey = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    
    n = l_partkey.numel()
    device = l_partkey.device

    # Accumulators for average quantity calculation
    sum_qty = torch.zeros(array_size, dtype=torch.float64, device=device)
    cnt_qty = torch.zeros(array_size, dtype=torch.float64, device=device)

    # Kernel 1: Accumulate sum and count per partkey for filtered parts
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._accumulate_kernel[grid](
        l_partkey, l_quantity, is_filtered,
        sum_qty, cnt_qty,
        n,
    )

    # Compute thresholds: 0.2 * avg_qty
    # Handle division by zero where cnt_qty == 0
    thresholds = torch.zeros(array_size, dtype=torch.float64, device=device)
    valid_cnt = cnt_qty > 0
    thresholds[valid_cnt] = 0.2 * sum_qty[valid_cnt] / cnt_qty[valid_cnt]

    # Final accumulator for revenue
    out = torch.zeros(1, dtype=torch.float64, device=device)

    # Kernel 2: Filter by threshold and sum extendedprice
    _K._filter_sum_kernel[grid](
        l_partkey, l_quantity, l_extendedprice, is_filtered, thresholds,
        out,
        n,
    )

    return float(out.item() / 7.0)
