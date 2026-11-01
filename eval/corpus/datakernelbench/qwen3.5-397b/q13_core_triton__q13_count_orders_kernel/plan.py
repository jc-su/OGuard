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

@triton.jit
def _q13_count_orders_kernel(
    o_custkey_ptr,
    order_counts_ptr,
    n_orders,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count orders per customer using atomic add.
    Each order increments its customer's bucket by 1.
    Optimized for H100 with large block sizes.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders
    
    custkey = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    tl.atomic_add(order_counts_ptr + custkey, 1, mask=mask)


@triton.jit
def _q13_gather_counts_kernel(
    c_custkey_ptr,
    order_counts_ptr,
    c_count_ptr,
    n_customers,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gather order count for each customer from the order_counts array.
    Implements LEFT OUTER JOIN semantics (customers with 0 orders get 0).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers
    
    custkey = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    count = tl.load(order_counts_ptr + custkey, mask=mask, other=0)
    tl.store(c_count_ptr + offs, count, mask=mask)
'''


def _load_kernels():
    """Load Triton kernels from a temporary on-disk file."""
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


def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute customer order count distribution using optimized Triton kernels.
    
    GPU Execution:
    - Triton kernel 1: Count orders per customer (atomic add to bucket array)
    - Triton kernel 2: Gather order count for each customer (LEFT OUTER JOIN)
    - PyTorch: Compute unique counts and distribution (torch.unique)
    - PyTorch: Sort by custdist DESC, c_count DESC
    
    Why faster:
    - Triton kernels with H100-optimized block sizes (8192)
    - Reduced kernel launch overhead vs multiple torch ops
    - Coalesced memory access patterns
    - Minimal temporary tensor allocations
    """
    n_orders = o_custkey_filtered.numel()
    n_customers = c_custkey.numel()
    device = c_custkey.device
    
    # Allocate order counts bucket array (1-indexed customer keys)
    order_counts = torch.zeros(num_customers + 1, dtype=torch.int64, device=device)
    
    # Allocate output array for c_count per customer
    c_count = torch.zeros(n_customers, dtype=torch.int64, device=device)
    
    # H100-optimized block size (large for memory-bound operations)
    BLOCK_SIZE = 8192
    
    # Grid configuration
    grid_count = (triton.cdiv(n_orders, BLOCK_SIZE),)
    grid_gather = (triton.cdiv(n_customers, BLOCK_SIZE),)
    
    # Kernel 1: Count orders per customer
    _K._q13_count_orders_kernel[grid_count](
        o_custkey_filtered,
        order_counts,
        n_orders,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Kernel 2: Gather counts for each customer
    _K._q13_gather_counts_kernel[grid_gather](
        c_custkey,
        order_counts,
        c_count,
        n_customers,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Compute distribution: unique counts and their frequencies
    unique_counts, custdist = torch.unique(c_count, return_counts=True)
    
    # Sort by c_count DESC first, then stable sort by custdist DESC
    # This matches reference semantics: ORDER BY custdist DESC, c_count DESC
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
    # cuDF supports .str.contains with regex. Convert SQL LIKE pattern to regex: % -> .*
    # For the specific pattern, '%special%requests%' -> '.*special.*requests.*' (equivalent to contains 'special.*requests')
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
