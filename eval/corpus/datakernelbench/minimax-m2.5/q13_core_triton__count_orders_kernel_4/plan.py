import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf

# ----------------------------------------------------------------------
# Triton kernel source – must reside on disk to be compiled by Triton.
# ----------------------------------------------------------------------
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
    key=["num_orders"],
    reset_to_zero=["order_counts_ptr"],
)
@triton.jit
def _count_orders_kernel(
    o_custkey_ptr,
    order_counts_ptr,
    num_orders,
    num_customers_plus_one,
    BLOCK_SIZE: tl.constexpr,
):
    """Count filtered orders per customer using atomic increments."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_orders

    custkey = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    tl.atomic_add(order_counts_ptr + custkey, 1)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q13_triton_")
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
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Step 1: For each customer, count matching orders (from filtered orders).
    # Use Triton kernel to count orders per customer.
    n_orders = o_custkey_filtered.numel()
    order_counts_per_cust = torch.zeros(
        num_customers + 1, dtype=torch.int64, device=c_custkey.device
    )
    grid = lambda meta: (triton.cdiv(n_orders, meta["BLOCK_SIZE"]),)
    _K._count_orders_kernel[grid](
        o_custkey_filtered, order_counts_per_cust, n_orders, num_customers + 1
    )

    # Gather the count for each c_custkey (LEFT OUTER JOIN => customers with 0 orders kept as 0)
    c_count = order_counts_per_cust[c_custkey.to(torch.int64)]

    # Step 2: Group by c_count -> custdist = count(*)
    # Use torch.unique which is highly optimized on GPU.
    unique_counts, custdist = torch.unique(c_count, return_counts=True)

    # Step 3: ORDER BY custdist DESC, c_count DESC
    order1 = torch.argsort(unique_counts, descending=True, stable=True)
    unique_counts = unique_counts[order1]
    custdist = custdist[order1]

    order2 = torch.argsort(custdist, descending=True, stable=True)
    unique_counts = unique_counts[order2]
    custdist = custdist[order2]

    return unique_counts, custdist


def run_query(
    tables: dict[str, cudf.DataFrame],
    comment_pattern: str = "%special%requests%",
) -> cudf.DataFrame:
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
    Description: TPC-H Q13 - Customer distribution based on number of
                 non‑special‑request orders.
    Params: comment_pattern (str): the LIKE pattern to exclude from orders
            via NOT LIKE.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Filter orders: o_comment NOT LIKE '%special%requests%'
    mask = ~df_orders["o_comment"].str.contains("special.*requests", regex=True)
    filtered_orders = df_orders[mask]

    # Prepare GPU tensors.
    c_custkey_t = torch.as_tensor(
        df_customer["c_custkey"].to_cupy(), dtype=torch.int64
    )
    o_custkey_t = torch.as_tensor(
        filtered_orders["o_custkey"].to_cupy(), dtype=torch.int64
    )

    # Upper bound for bucket size – max customer key.
    num_customers = int(df_customer["c_custkey"].max())

    # Run the Triton‑accelerated core.
    c_count_t, custdist_t = _query_core(c_custkey_t, o_custkey_t, num_customers)

    return cudf.DataFrame({
        "c_count": c_count_t,
        "custdist": custdist_t,
    })
