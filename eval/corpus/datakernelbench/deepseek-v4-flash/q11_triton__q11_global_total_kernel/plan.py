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

@triton.jit
def _q11_global_total_kernel(
    supplycost_ptr, availqty_ptr, total_out, n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    availqty = tl.load(availqty_ptr + offs, mask=mask, other=0.0)
    value = supplycost * availqty
    block_sum = tl.sum(tl.where(mask, value, 0.0))
    tl.atomic_add(total_out, block_sum)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q11_triton_")
    path = os.path.join(tmpdir, "_q11_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q11_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated TPC-H Q11 aggregation using PyTorch primitives.
    
    Steps:
    1. Compute value = supplycost * availqty
    2. Compute global total via Triton kernel
    3. Group by partkey using unique + scatter_add
    4. Filter by threshold
    5. Sort by value DESC
    """
    n = ps_partkey.numel()
    device = ps_partkey.device
    
    # Step 1: Compute value
    value = ps_supplycost * ps_availqty
    
    # Step 2: Compute global total using Triton kernel (no autotune to avoid conflicts)
    total_out = torch.zeros(1, dtype=torch.float64, device=device)
    BLOCK_SIZE = 4096
    num_warps = 4
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _K._q11_global_total_kernel[grid](
        ps_supplycost, ps_availqty, total_out, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    total_val = total_out.item()
    threshold = total_val * threshold_fraction
    
    # Step 3: Group by partkey using unique + scatter_add
    unique_keys, inverse = torch.unique(ps_partkey, return_inverse=True)
    n_groups = unique_keys.shape[0]
    
    grouped_values = torch.zeros(n_groups, dtype=torch.float64, device=device)
    grouped_values.scatter_add_(0, inverse, value)
    
    # Step 4: Filter by threshold
    mask = grouped_values > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = grouped_values[mask]
    
    # Step 5: Sort by value DESC
    sort_idx = torch.argsort(filtered_values, descending=True)
    
    return filtered_keys[sort_idx], filtered_values[sort_idx]


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "GERMANY",
    threshold_fraction: float = 0.0001,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        GROUP BY ps_partkey
        HAVING sum(ps_supplycost * ps_availqty) > (
            SELECT sum(ps_supplycost * ps_availqty) * 0.0001
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        )
        ORDER BY value DESC;
    Description: TPC-H Q11 implementation.
    - cuDF handles the joins (nation → supplier → partsupp), utilizing its efficient GPU join 
      implementations for the small dimension tables.
    - GPU (PyTorch + Triton) accelerates the aggregation hot path: value computation, global total,
      group-by sum, threshold filter, and sort.
    - Uses PyTorch's scatter_add for group-by aggregation (highly optimized on GPU).
    - Uses a simple Triton kernel (without autotune) for computing the global total 
      (memory-bound sum across all rows).
    Params:
        nation_name (str): nation name filter literal.
        threshold_fraction (float): fractional threshold used in HAVING clause.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    
    # Filter nation
    filtered_nation = df_nation[df_nation["n_name"] == nation_name]
    
    # Join supplier <-> nation
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )
    
    # Join partsupp <-> (supplier <-> nation)
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")
    
    # Convert to GPU tensors
    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)
    
    # Compute using optimized GPU path
    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )
    
    return cudf.DataFrame({
        "ps_partkey": keys_t.cpu(),
        "value": values_t.cpu(),
    })
