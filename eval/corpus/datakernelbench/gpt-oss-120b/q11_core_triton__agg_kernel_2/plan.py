import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source – written to a temporary file and imported.
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune configurations for Hopper (H100)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=wp, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
        for wp in [4, 8]
    ],
    key=["n_elements"],
    reset_to_zero=["total_ptr"],   # ensure the atomic accumulator starts at zero
)
@triton.jit
def _agg_kernel(
    supply_ptr,          # float64 *
    avail_ptr,           # float64 *
    out_ptr,             # float64 *
    total_ptr,           # float64 *
    n_elements,          # i64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Valid‑index mask
    mask = offs < n_elements

    # Load inputs; out‑of‑bounds lanes receive 0.0
    supply = tl.load(supply_ptr + offs, mask=mask, other=0.0)
    avail  = tl.load(avail_ptr  + offs, mask=mask, other=0.0)

    # Per‑row product
    val = supply * avail

    # Write product to output buffer
    tl.store(out_ptr + offs, val, mask=mask)

    # Zero OOB lanes before reduction
    val_masked = tl.where(mask, val, 0.0)

    # Block‑wise sum then atomic add to global total
    block_sum = tl.sum(val_masked)
    tl.atomic_add(total_ptr, block_sum)
'''

def _load_kernels():
    """Write the Triton source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q11_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels once at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Private helper that launches the fused aggregation kernel.
# ----------------------------------------------------------------------
def _launch_agg(
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per‑row ``ps_supplycost * ps_availqty`` and its global sum
    using a single Triton kernel launch.

    Returns
    -------
    values : torch.Tensor
        Per‑row product, same shape and dtype as inputs.
    total : torch.Tensor
        1‑element tensor containing the sum of all ``values`` (float64).
    """
    n = ps_supplycost.numel()
    values = torch.empty_like(ps_supplycost, device=ps_supplycost.device)
    total = torch.zeros(1, dtype=ps_supplycost.dtype, device=ps_supplycost.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._agg_kernel[grid](
        ps_supplycost,
        ps_availqty,
        values,
        total,
        n,
    )
    return values, total

# ----------------------------------------------------------------------
# Public compute core – signature matches the reference exactly.
# ----------------------------------------------------------------------
def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton‑fused multiplication and global sum, followed by torch‑based
    grouping, HAVING filter and ORDER BY.

    The heavy per‑row product and total aggregation are performed in a
    single memory‑bound Triton kernel, reducing HBM traffic compared with the
    original eager PyTorch implementation.
    """
    # 1) Compute per‑row product and global total via the fused kernel.
    values, total = _launch_agg(ps_supplycost, ps_availqty)

    # 2) GROUP BY ps_partkey.
    unique_keys, inverse = torch.unique(ps_partkey, return_inverse=True)
    grouped_values = torch.zeros_like(
        unique_keys, dtype=values.dtype, device=values.device
    )
    grouped_values.scatter_add_(0, inverse, values)

    # 3) HAVING filter based on the fraction of the global total.
    threshold = total * threshold_fraction
    mask = grouped_values > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = grouped_values[mask]

    # 4) ORDER BY value DESC.
    sort_idx = torch.argsort(filtered_values, descending=True)
    return filtered_keys[sort_idx], filtered_values[sort_idx]

# ----------------------------------------------------------------------
# run_query – unchanged from the reference implementation.
# ----------------------------------------------------------------------
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
    ps_supplycost_t = torch.as_tensor(
        joined["ps_supplycost"].to_cupy(), dtype=torch.float64
    )
    ps_availqty_t = torch.as_tensor(
        joined["ps_availqty"].to_cupy(), dtype=torch.float64
    )

    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    return cudf.DataFrame({
        "ps_partkey": keys_t,
        "value": values_t,
    })
