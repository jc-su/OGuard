import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [4096, 8192, 16384]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q17_filter_sum_kernel(
    l_price_ptr, keep_mask_ptr,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keep = tl.load(keep_mask_ptr + offs, mask=mask, other=0)
    price = tl.load(l_price_ptr + offs, mask=keep.to(tl.int1), other=0.0)
    
    block_sum = tl.sum(price)
    tl.atomic_add(out_ptr, block_sum)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q17_triton_")
    path = os.path.join(tmpdir, "_q17_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q17_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC-H Q17: sum(l_extendedprice)/7.0 where l_quantity < 0.2 * avg(l_quantity) over parts
    matching brand/container filters, grouped by partkey.
    
    This version uses PyTorch for the membership check and per-part averages,
    then only the final filter+aggregate is done in a fused Triton kernel.
    """
    device = l_partkey.device
    
    # Sort part_keys once on GPU
    sorted_keys, _ = torch.sort(part_keys)
    n_parts = sorted_keys.numel()
    
    # For each lineitem, find index in sorted_keys using searchsorted (fast PyTorch)
    idx = torch.searchsorted(sorted_keys, l_partkey)
    idx_clamped = torch.clamp(idx, max=n_parts - 1)
    in_part = sorted_keys[idx_clamped] == l_partkey
    
    # Compute per-part average quantity
    sel_idx = idx_clamped[in_part]
    sel_qty = l_quantity[in_part]
    ones = torch.ones_like(sel_qty)
    
    sum_qty = torch.zeros(n_parts, dtype=torch.float64, device=device)
    cnt_qty = torch.zeros(n_parts, dtype=torch.float64, device=device)
    sum_qty.scatter_add_(0, sel_idx, sel_qty)
    cnt_qty.scatter_add_(0, sel_idx, ones)
    avg_qty = torch.where(cnt_qty > 0, sum_qty / cnt_qty, torch.zeros_like(sum_qty))
    
    # Compute threshold and mask for all lineitems
    # Use float64 for precision
    line_avg_qty = avg_qty[idx_clamped]
    threshold = 0.2 * line_avg_qty
    keep_mask = in_part & (l_quantity < threshold)
    
    # Count rows and early exit if none
    n_keep = keep_mask.sum()
    if n_keep == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=device)
    
    # Fused Triton kernel for final sum
    n = l_extendedprice.numel()
    out = torch.zeros(1, dtype=torch.float64, device=device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q17_filter_sum_kernel[grid](
        l_extendedprice, keep_mask,
        out, n,
    )
    return out.squeeze(0) / 7.0


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
        given brand and container. Correlated subquery is evaluated as a grouped average over
        all lineitems for each filtered partkey.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table first (cuDF)
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    # Move to GPU tensors
    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
