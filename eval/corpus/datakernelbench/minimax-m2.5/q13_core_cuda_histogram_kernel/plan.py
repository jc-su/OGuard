import os
import tempfile
import torch
from torch.utils.cpp_extension import load
import cudf

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void histogram_kernel(
    const int64_t* __restrict__ keys,
    int64_t n,
    int64_t* __restrict__ counts
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        atomicAdd((unsigned long long *)&counts[key], 1ULL);
    }
}

torch::Tensor histogram_add(torch::Tensor keys, torch::Tensor counts) {
    TORCH_CHECK(keys.scalar_type() == torch::kInt64, "keys must be int64");
    TORCH_CHECK(counts.scalar_type() == torch::kInt64, "counts must be int64");

    if (!keys.is_cuda()) keys = keys.cuda();
    if (!counts.is_cuda()) counts = counts.cuda();

    keys = keys.contiguous();
    counts = counts.contiguous();

    int64_t n = keys.numel();
    if (n == 0) return counts;

    int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogram_kernel<<<blocks, BLOCK_SIZE>>>(
        keys.data_ptr<int64_t>(),
        n,
        counts.data_ptr<int64_t>()
    );
    cudaDeviceSynchronize();
    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("histogram_add", &histogram_add, "Histogram add (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q13_cuda_")
    cu_path = os.path.join(tmpdir, "_q13_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q13_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the distribution of order counts per customer using a CUDA histogram
    kernel for the hot path (counting orders per customer). The remaining steps
    (grouping, sorting) use standard PyTorch operations.
    """
    # Allocate histogram bucket: index 0 unused, customers are 1-indexed.
    order_counts_per_cust = torch.zeros(
        num_customers + 1, dtype=torch.int64, device=c_custkey.device
    )

    if o_custkey_filtered.numel() > 0:
        # Fused CUDA kernel for histogram (fast scatter-add)
        order_counts_per_cust = _EXT.histogram_add(
            o_custkey_filtered, order_counts_per_cust
        )

    # Gather the count for each customer (LEFT OUTER JOIN semantics)
    c_count = order_counts_per_cust[c_custkey.to(torch.int64)]

    # Group by c_count -> custdist = count(*)
    unique_counts, custdist = torch.unique(c_count, return_counts=True)

    # ORDER BY custdist DESC, c_count DESC (stable sorts to match reference)
    order1 = torch.argsort(unique_counts, descending=True, stable=True)
    unique_counts = unique_counts[order1]
    custdist = custdist[order1]

    order2 = torch.argsort(custdist, descending=True, stable=True)
    unique_counts = unique_counts[order2]
    custdist = custdist[order2]

    return unique_counts, custdist

def run_query(
    tables: dict[str, cudf.DataFrame],
    comment_pattern: str = "%special%requests%"
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
