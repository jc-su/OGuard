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
def _q13_scatter_add_kernel(
    o_custkey_ptr,
    order_counts_ptr,
    n_orders,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders
    
    o_keys = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    valid = mask & (o_keys > 0)
    
    # Use tl.atomic_add with block-level offsets
    # Cast to int64 for indexing
    o_keys_int = tl.cast(o_keys, tl.int64)
    ones = tl.where(valid, tl.full((BLOCK_SIZE,), 1, tl.int64), tl.zeros((BLOCK_SIZE,), tl.int64))
    tl.atomic_add(order_counts_ptr + o_keys_int, ones, mask=valid)

@triton.jit
def _q13_gather_counts_kernel(
    c_custkey_ptr,
    order_counts_ptr,
    c_count_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    c_keys = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    counts = tl.load(order_counts_ptr + c_keys, mask=mask, other=0)
    
    tl.store(c_count_out_ptr + offs, counts, mask=mask)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q13_triton_")
    path = os.path.join(tmpdir, "_q13_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q13_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _launch_scatter_add(
    o_custkey: torch.Tensor,
    num_customers: int,
) -> torch.Tensor:
    n = o_custkey.numel()
    order_counts = torch.zeros(num_customers + 1, dtype=torch.int64, device=o_custkey.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _K._q13_scatter_add_kernel[grid](
        o_custkey,
        order_counts,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return order_counts

def _launch_gather(
    c_custkey: torch.Tensor,
    order_counts: torch.Tensor,
) -> torch.Tensor:
    n = c_custkey.numel()
    c_count = torch.zeros(n, dtype=torch.int64, device=c_custkey.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _K._q13_gather_counts_kernel[grid](
        c_custkey,
        order_counts,
        c_count,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return c_count

def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Triton kernel plan for TPC-H Q13.
    
    What runs where:
    - _launch_scatter_add: Triton kernel performs atomic scatter-add (GPU)
    - _launch_gather: Triton kernel performs gather of counts (GPU)
    - torch.unique: PyTorch built-in for final group-by (GPU)
    
    Why faster: Two fused Triton kernels replace separate group-by and join operations.
    Atomic scatter-add avoids materializing intermediate join results.
    """
    # Step 1: For each customer, count matching orders via atomic scatter-add
    order_counts = _launch_scatter_add(o_custkey_filtered, num_customers)
    
    # Step 2: Gather the count for each c_custkey
    c_count = _launch_gather(c_custkey, order_counts)
    
    # Step 3: Group by c_count -> custdist = count(*)
    unique_counts, custdist = torch.unique(c_count, return_counts=True)
    
    # Step 4: ORDER BY custdist DESC, c_count DESC
    order1 = torch.argsort(unique_counts, descending=True, stable=True)
    unique_counts = unique_counts[order1]
    custdist = custdist[order1]
    order2 = torch.argsort(custdist, descending=True, stable=True)
    unique_counts = unique_counts[order2]
    custdist = custdist[order2]
    
    return unique_counts, custdist


def run_query(tables: dict[str, cudf.DataFrame], comment_pattern: str = "%special%requests%") -> cudf.DataFrame:
    """
    SQL:
        SELECT c_count, count(*) AS custdist
        FROM (
            SELECT c_custkey, count(o_orderkey)
            FROM customer LEFT OUTER JOIN orders
              ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'
            GROUP BY c_custkey
        ) AS c_orders (c_custkey, c_count)
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;
    Description: TPC-H Q13 - Customer distribution based on number of non-special-request orders.
    Params: comment_pattern (str): the LIKE pattern to exclude from orders via NOT LIKE.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Filter orders: o_comment NOT LIKE '%special%requests%'
    mask = ~df_orders["o_comment"].str.contains("special.*requests", regex=True)
    filtered_orders = df_orders[mask]

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    o_custkey_t = torch.as_tensor(filtered_orders["o_custkey"].to_cupy(), dtype=torch.int64)

    # Use max custkey as the bucket size upper bound
    num_customers = int(df_customer["c_custkey"].max())

    c_count_t, custdist_t = _query_core(c_custkey_t, o_custkey_t, num_customers)

    return cudf.DataFrame({
        "c_count": c_count_t,
        "custdist": custdist_t,
    })
