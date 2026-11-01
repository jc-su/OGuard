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
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q11_accumulate_kernel(
    supplycost_ptr, availqty_ptr, inverse_ptr,
    out_ptr, n_elements, n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q11: compute value = supplycost * availqty and
    scatter_add into grouped output by inverse indices.
    
    Memory-bound optimization: single pass through data, computes value in registers,
    atomic_add directly to grouped output. Avoids separate multiply + scatter passes.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load data with safe defaults for OOB lanes
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    availqty = tl.load(availqty_ptr + offs, mask=mask, other=0.0)
    inverse = tl.load(inverse_ptr + offs, mask=mask, other=0)
    
    # Compute value in registers (no extra HBM traffic)
    value = supplycost * availqty
    
    # Atomic add to grouped output at inverse[group_idx]
    tl.atomic_add(out_ptr + inverse, value, mask=mask)
'''


def _load_kernels():
    """
    Load Triton kernels from a temp file.
    
    Required pattern: Triton JIT kernels must be in a separate on-disk file
    for the validator's import mechanism to work correctly.
    """
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


def _launch_q11_accumulate(
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    inverse: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """
    Launch Triton kernel to compute value = supplycost * availqty and
    accumulate into grouped output using inverse indices from torch.unique.
    
    GPU: Triton fused kernel (compute + scatter_add in one pass)
    Why faster: Single kernel launch, value computed in registers, direct atomic_add
    to grouped output avoids intermediate tensor allocations.
    """
    n_elements = ps_supplycost.numel()
    out = torch.zeros(n_groups, dtype=torch.float64, device=ps_supplycost.device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _K._q11_accumulate_kernel[grid](
        ps_supplycost,
        ps_availqty,
        inverse,
        out,
        n_elements,
        n_groups,
    )
    return out


def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core query logic for TPC-H Q11 using Triton for fused value computation + aggregation.
    
    GPU: Triton fused kernel for value computation and group accumulation
    Why faster: Single kernel pass computes value and scatters to groups,
    avoiding separate multiply + scatter_add kernel launches.
    
    Args:
        ps_partkey: Part key tensor (int64)
        ps_supplycost: Supply cost tensor (float64)
        ps_availqty: Available quantity tensor (float64)
        threshold_fraction: Fraction of total for HAVING clause
    
    Returns:
        Tuple of (filtered_keys, filtered_values) sorted by value DESC
    """
    # Get unique partkeys and inverse mapping for groupby
    unique_keys, inverse = torch.unique(ps_partkey, return_inverse=True)
    n_groups = unique_keys.shape[0]
    
    # Fused Triton kernel: compute value and scatter_add in one pass
    grouped_values = _launch_q11_accumulate(
        ps_supplycost, ps_availqty, inverse, n_groups
    )
    
    # Global threshold computation
    total = torch.sum(grouped_values)
    threshold = total * threshold_fraction
    
    # HAVING filter: keep groups where sum > threshold
    mask = grouped_values > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = grouped_values[mask]
    
    # ORDER BY value DESC
    if filtered_values.numel() > 0:
        sort_idx = torch.argsort(filtered_values, descending=True)
        return filtered_keys[sort_idx], filtered_values[sort_idx]
    else:
        return filtered_keys, filtered_values


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
    
    Description:
        TPC-H Q11 - Identifies parts contributing significantly to the total
        supply cost value for suppliers in a given nation.
        
        Data flow:
        - cuDF: Joins (partsupp ⋈ supplier ⋈ nation), string filtering (n_name)
        - GPU (Triton): Fused kernel computes value = supplycost * availqty and
          accumulates into grouped output in a single pass, avoiding separate
          multiply + scatter_add kernel launches. This reduces HBM traffic and
          kernel launch overhead for the memory-bound column scan.
        - PyTorch: Threshold computation, HAVING filter, sorting (small result set)
    
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables (partsupp, supplier, nation)
        nation_name (str): Nation name filter literal (default: 'GERMANY')
        threshold_fraction (float): Fractional threshold for HAVING clause (default: 0.0001)
    
    Returns:
        cudf.DataFrame: Columns [ps_partkey, value] sorted by value DESC
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    # Filter nation by name (string comparison on CPU/cuDF)
    filtered_nation = df_nation[df_nation["n_name"] == nation_name]

    # Join supplier <-> nation on s_nationkey = n_nationkey
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )

    # Join partsupp <-> (supplier ⋈ nation) on ps_suppkey = s_suppkey
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")

    # Convert to GPU tensors with explicit dtypes
    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)

    # Run optimized Triton-based query core
    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    # Return as cudf.DataFrame with correct column names
    return cudf.DataFrame({
        "ps_partkey": keys_t,
        "value": values_t,
    })
