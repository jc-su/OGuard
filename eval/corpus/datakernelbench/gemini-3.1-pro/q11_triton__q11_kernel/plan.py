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
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
    ],
    key=["n_elements"],
    reset_to_zero=["out_sums_ptr"],
)
@triton.jit
def _q11_kernel(
    ps_suppkey_ptr,
    ps_partkey_ptr,
    ps_supplycost_ptr,
    ps_availqty_ptr,
    is_nation_supp_ptr,
    out_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load supplier key and check validity (is it in the target nation?)
    suppkey = tl.load(ps_suppkey_ptr + offs, mask=mask, other=0)
    valid = tl.load(is_nation_supp_ptr + suppkey, mask=mask, other=0)
    keep = mask & (valid == 1)

    # Conditionally load heavy columns (low selectivity on valid suppliers)
    partkey = tl.load(ps_partkey_ptr + offs, mask=keep, other=0)
    cost = tl.load(ps_supplycost_ptr + offs, mask=keep, other=0.0)
    qty = tl.load(ps_availqty_ptr + offs, mask=keep, other=0.0)

    val = cost * qty

    # Atomic add to the group's slot in out_sums array
    tl.atomic_add(out_sums_ptr + partkey, val, mask=keep)
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
        Optimized execution using a fused Triton kernel.
        Instead of materializing large joins for the partsupp table in cuDF, 
        we find the valid supplier keys natively via cuDF. Then we use those 
        keys to populate an O(1) direct-lookup hash array (since IDs are integers). 
        The Triton kernel scans the partsupp table in a single memory-bound pass, 
        checks supplier validity in registers, and atomic-scatters the value into 
        a direct array sized by max(ps_partkey). The total sum is then accumulated 
        host-side over the scatter array for the HAVING threshold.
        
    Params:
        nation_name (str): nation name filter literal.
        threshold_fraction (float): fractional threshold used in HAVING clause.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    if len(df_partsupp) == 0:
        return cudf.DataFrame({
            "ps_partkey": pd.Series(dtype="int64"),
            "value": pd.Series(dtype="float64")
        })

    # 1. cuDF: find valid suppliers in the target nation
    filtered_nation = df_nation[df_nation["n_name"] == nation_name]
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )
    
    valid_suppkeys = sup_nat["s_suppkey"].to_cupy()
    valid_suppkeys_t = torch.as_tensor(valid_suppkeys, dtype=torch.int64)

    # 2. Extract tensors for the primary heavy table
    suppkey_t = torch.as_tensor(df_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    partkey_t = torch.as_tensor(df_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    cost_t = torch.as_tensor(df_partsupp["ps_supplycost"].to_cupy(), dtype=torch.float64)
    qty_t = torch.as_tensor(df_partsupp["ps_availqty"].to_cupy(), dtype=torch.float64)

    # 3. Create O(1) lookup map for valid suppliers
    max_suppkey_df = int(suppkey_t.max().item())
    max_suppkey_valid = int(valid_suppkeys_t.max().item()) if valid_suppkeys_t.numel() > 0 else 0
    max_suppkey = max(max_suppkey_df, max_suppkey_valid)
    
    is_nation_supp_t = torch.zeros(max_suppkey + 1, dtype=torch.int32, device=suppkey_t.device)
    if valid_suppkeys_t.numel() > 0:
        is_nation_supp_t[valid_suppkeys_t] = 1

    # 4. Prepare array for direct partkey accumulation
    max_partkey = int(partkey_t.max().item())
    out_sums_t = torch.zeros(max_partkey + 1, dtype=torch.float64, device=partkey_t.device)
    
    # 5. Launch Fused Triton Kernel
    n_elements = suppkey_t.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _K._q11_kernel[grid](
        suppkey_t,
        partkey_t,
        cost_t,
        qty_t,
        is_nation_supp_t,
        out_sums_t,
        n_elements
    )

    # 6. Post-processing: Global Aggregation, HAVING filter, and ORDER BY
    total_val = out_sums_t.sum()
    threshold = total_val * threshold_fraction
    
    # Find groups that pass the HAVING threshold
    mask = out_sums_t > threshold
    filtered_keys = torch.nonzero(mask, as_tuple=False).squeeze(1)
    filtered_values = out_sums_t[filtered_keys]

    # ORDER BY value DESC
    sort_idx = torch.argsort(filtered_values, descending=True)
    sorted_keys = filtered_keys[sort_idx]
    sorted_values = filtered_values[sort_idx]

    return cudf.DataFrame({
        "ps_partkey": sorted_keys,
        "value": sorted_values,
    })
