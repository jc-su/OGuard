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
)
@triton.jit
def pass1_kernel(
    l_partkey_ptr, l_quantity_ptr,
    is_valid_ptr, sum_qty_ptr, cnt_qty_ptr,
    max_pk, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load partkey, default to 0 for out-of-bounds lanes
    pk = tl.load(l_partkey_ptr + offs, mask=mask, other=0)
    
    # Ensure safe indexing into the dense lookup tables
    valid_idx = (pk >= 0) & (pk <= max_pk)
    safe_pk = tl.where(valid_idx, pk, 0)
    
    # Check if this partkey is one of the filtered parts
    is_valid = tl.load(is_valid_ptr + safe_pk, mask=mask & valid_idx, other=0)
    keep = mask & valid_idx & (is_valid == 1)
    
    # Load quantity only for valid items to save bandwidth
    qty = tl.load(l_quantity_ptr + offs, mask=keep, other=0.0).to(tl.float64)
    one = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float64)
    
    # Atomically accumulate sum and count directly to HBM
    tl.atomic_add(sum_qty_ptr + safe_pk, qty, mask=keep)
    tl.atomic_add(cnt_qty_ptr + safe_pk, one, mask=keep)


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
def pass2_kernel(
    l_partkey_ptr, l_quantity_ptr, l_extendedprice_ptr,
    is_valid_ptr, threshold_ptr, out_ptr,
    max_pk, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load partkey, default to 0 for out-of-bounds lanes
    pk = tl.load(l_partkey_ptr + offs, mask=mask, other=0)
    
    valid_idx = (pk >= 0) & (pk <= max_pk)
    safe_pk = tl.where(valid_idx, pk, 0)
    
    # Only process partkeys in our valid set
    is_valid = tl.load(is_valid_ptr + safe_pk, mask=mask & valid_idx, other=0)
    keep_part = mask & valid_idx & (is_valid == 1)
    
    # Load quantity and pre-computed threshold
    qty = tl.load(l_quantity_ptr + offs, mask=keep_part, other=0.0).to(tl.float64)
    thresh = tl.load(threshold_ptr + safe_pk, mask=keep_part, other=0.0).to(tl.float64)
    
    # Apply threshold filter logic
    keep = keep_part & (qty < thresh)
    
    # Load extended price only for rows satisfying the subquery
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0).to(tl.float64)
    
    # Block-local reduce followed by global atomic add
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, price, 0.0)))
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
    Computes Q17 aggregations using 2 Fused Triton passes over the lineitem table.
    We convert O(log N) binary search into O(1) dense tensor mapping.
    """
    if part_keys.numel() == 0 or l_partkey.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=l_partkey.device)
        
    # Get max partkey to allocate dense lookup tensors
    max_pk = int(max(part_keys.max().item(), l_partkey.max().item()))
    
    # O(1) membership lookup: is_valid[partkey]
    is_valid = torch.zeros(max_pk + 1, dtype=torch.uint8, device=l_partkey.device)
    is_valid[part_keys] = 1
    
    # Per-part aggregations
    sum_qty = torch.zeros(max_pk + 1, dtype=torch.float64, device=l_partkey.device)
    cnt_qty = torch.zeros(max_pk + 1, dtype=torch.float64, device=l_partkey.device)
    
    n_elements = l_partkey.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Pass 1: Sum quantities for valid partkeys
    _K.pass1_kernel[grid](
        l_partkey, l_quantity,
        is_valid, sum_qty, cnt_qty,
        max_pk, n_elements
    )
    
    # Compute threshold = 0.2 * avg(l_quantity) per partkey
    avg_qty = torch.where(
        cnt_qty > 0, 
        sum_qty / cnt_qty, 
        torch.tensor(0.0, dtype=torch.float64, device=cnt_qty.device)
    )
    threshold = 0.2 * avg_qty
    
    # Global output revenue accumulator
    out = torch.zeros(1, dtype=torch.float64, device=l_partkey.device)
    
    # Pass 2: Filter by threshold and aggregate extended price
    _K.pass2_kernel[grid](
        l_partkey, l_quantity, l_extendedprice,
        is_valid, threshold, out,
        max_pk, n_elements
    )
    
    # Apply divisor from query projection
    return out[0] / 7.0


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
        given brand and container. Uses dense boolean indexing and two fused Triton passes over lineitem.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table first (cuDF)
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]
    
    if len(df_part_f) == 0:
        return 0.0

    # Tensorize relevant columns to PyTorch
    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    # Execute fused core query
    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
