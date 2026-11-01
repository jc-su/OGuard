import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf

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
    reset_to_zero=["grouped_ptr", "total_ptr"],
)
@triton.jit
def _grouped_sum_kernel(
    supplycost_ptr,
    availqty_ptr,
    inverse_ptr,
    grouped_ptr,
    total_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load supplycost and availqty (float64)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    availqty = tl.load(availqty_ptr + offs, mask=mask, other=0.0)

    # Compute product (value) in registers
    product = supplycost * availqty

    # Add block-level sum to global total
    tl.atomic_add(total_ptr, tl.sum(product))

    # Load group indices (inverse mapping)
    inv = tl.load(inverse_ptr + offs, mask=mask, other=0)

    # Atomically add each row's product to its group accumulator
    tl.atomic_add(grouped_ptr + inv, product)
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


def _launch_grouped_sum(
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    inverse: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch fused Triton kernel to compute per‑partkey sums and total sum."""
    n = ps_supplycost.numel()
    device = ps_supplycost.device
    grouped = torch.zeros(num_groups, dtype=torch.float64, device=device)
    total = torch.zeros(1, dtype=torch.float64, device=device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._grouped_sum_kernel[grid](
        ps_supplycost,
        ps_availqty,
        inverse,
        grouped,
        total,
        n,
    )
    return grouped, total.squeeze(0)


def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute group‑by sum, apply HAVING threshold, and order by value descending."""
    # Unique part keys and inverse mapping (same as reference)
    unique_keys, inverse = torch.unique(ps_partkey, return_inverse=True)
    num_groups = unique_keys.shape[0]

    # Fused Triton kernel computes both per‑key sums and global total in one pass
    grouped_values, total_sum = _launch_grouped_sum(
        ps_supplycost, ps_availqty, inverse, num_groups
    )

    # Compute threshold from total sum
    threshold = total_sum.item() * threshold_fraction

    # HAVING filter: keep groups whose sum exceeds the threshold
    mask = grouped_values > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = grouped_values[mask]

    # ORDER BY value DESC
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
