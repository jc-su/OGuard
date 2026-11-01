import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import triton

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
def fused_scatter_add_kernel(
    cost_ptr, qty_ptr, inverse_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load inputs with mask to handle OOB safely
    c = tl.load(cost_ptr + offs, mask=mask, other=0.0)
    q = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    inv = tl.load(inverse_ptr + offs, mask=mask, other=0)
    
    # Compute value per row
    val = c * q
    
    # Scatter-add into group_sums buffer using inverse indices
    # mask ensures OOB threads do not perform atomic_add
    tl.atomic_add(out_ptr + inv, val, mask=mask)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q11_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
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
    Optimized core using Triton for fused value computation and scatter-add aggregation.
    Reduces memory traffic by avoiding intermediate 'value' tensor allocation and fusing
    the multiply and aggregate steps into a single kernel launch.
    """
    # Group by preparation: unique keys and inverse mapping
    # torch.unique is efficient and necessary to map sparse partkeys to dense indices
    unique_keys, inverse = torch.unique(ps_partkey, return_inverse=True)
    n_unique = unique_keys.shape[0]
    n_elements = ps_partkey.numel()
    
    # Output buffer for group sums (initialized to zero)
    group_sums = torch.zeros(n_unique, dtype=torch.float64, device=ps_partkey.device)
    
    # Launch fused kernel: value = cost * qty; group_sums[inverse] += value
    # This replaces: value = cost * qty; group_sums.scatter_add_(0, inverse, value)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _K.fused_scatter_add_kernel[grid](
        ps_supplycost, ps_availqty, inverse, group_sums,
        n_elements,
    )
    
    # Global threshold calculation (sum of all group sums)
    total = torch.sum(group_sums)
    threshold = total * threshold_fraction
    
    # HAVING filter: keep groups where sum > threshold
    mask = group_sums > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = group_sums[mask]
    
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
    Description: TPC-H Q11 - Identifies parts contributing significantly to the total
                 supply cost value for suppliers in a given nation.
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

    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)

    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    return cudf.DataFrame({
        "ps_partkey": keys_t,
        "value": values_t,
    })
