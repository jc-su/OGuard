import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

// Kernel: Count orders per customer using atomicAdd with proper casting
__global__ void count_orders_kernel(
    const int64_t* __restrict__ order_custkeys,
    int64_t* __restrict__ histogram,
    int64_t n_orders
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t custkey = order_custkeys[idx];
        atomicAdd(reinterpret_cast<unsigned long long*>(&histogram[custkey]), 1ULL);
    }
}

// Kernel: Gather counts for each customer (indexed read)
__global__ void gather_counts_kernel(
    const int64_t* __restrict__ custkeys,
    const int64_t* __restrict__ histogram,
    int64_t* __restrict__ counts,
    int64_t n_customers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_customers) {
        int64_t custkey = custkeys[idx];
        counts[idx] = histogram[custkey];
    }
}

// Launcher for order counting
void count_orders_launcher(
    torch::Tensor order_custkeys,
    torch::Tensor histogram,
    int64_t n_orders
) {
    TORCH_CHECK(order_custkeys.is_cuda());
    TORCH_CHECK(histogram.is_cuda());
    TORCH_CHECK(order_custkeys.scalar_type() == torch::kInt64);
    TORCH_CHECK(histogram.scalar_type() == torch::kInt64);

    int64_t n = n_orders;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_orders_kernel<<<blocks, BLOCK_SIZE>>>(
        order_custkeys.data_ptr<int64_t>(),
        histogram.data_ptr<int64_t>(),
        n
    );
}

// Launcher for gathering counts
void gather_counts_launcher(
    torch::Tensor custkeys,
    torch::Tensor histogram,
    torch::Tensor counts,
    int64_t n_customers
) {
    TORCH_CHECK(custkeys.is_cuda());
    TORCH_CHECK(histogram.is_cuda());
    TORCH_CHECK(counts.is_cuda());
    TORCH_CHECK(custkeys.scalar_type() == torch::kInt64);
    TORCH_CHECK(histogram.scalar_type() == torch::kInt64);
    TORCH_CHECK(counts.scalar_type() == torch::kInt64);

    int64_t n = n_customers;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gather_counts_kernel<<<blocks, BLOCK_SIZE>>>(
        custkeys.data_ptr<int64_t>(),
        histogram.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_orders", &count_orders_launcher, "Count orders per customer (CUDA)");
    m.def("gather_counts", &gather_counts_launcher, "Gather counts for each customer (CUDA)");
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
    Compute customer order count distribution using CUDA kernels.
    Step 1: Count orders per customer using atomicAdd in kernel.
    Step 2: Gather counts for each customer (left outer join).
    Step 3: Group by count using torch.unique.
    Step 4: Sort by custdist DESC, then c_count DESC.
    """
    device = c_custkey.device
    
    # Step 1: Create histogram (count per customer) and fill using CUDA kernel
    # histogram size: num_customers + 1 because customer keys start at 1
    histogram = torch.zeros(num_customers + 1, dtype=torch.int64, device=device)
    
    # Launch kernel to count orders per customer
    _EXT.count_orders(
        o_custkey_filtered,
        histogram,
        o_custkey_filtered.size(0)
    )
    
    # Step 2: Gather counts for each customer in c_custkey (left outer join)
    c_count = torch.empty_like(c_custkey)
    _EXT.gather_counts(
        c_custkey,
        histogram,
        c_count,
        c_custkey.size(0)
    )
    
    # Step 3: Group by c_count -> custdist = count(*)
    unique_counts, custdist = torch.unique(c_count, return_counts=True)
    
    # Step 4: ORDER BY custdist DESC, c_count DESC
    # Sort by c_count DESC first, then stable sort by custdist DESC.
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
                 Uses cuDF for string filtering and CUDA kernels for counting/gathering.
    Params: comment_pattern (str): the LIKE pattern to exclude from orders via NOT LIKE.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Filter orders: o_comment NOT LIKE '%special%requests%'
    mask = ~df_orders["o_comment"].str.contains("special.*requests", regex=True)
    filtered_orders = df_orders[mask]

    # Convert to tensors - ensure they're on CUDA and contiguous
    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64).cuda()
    o_custkey_t = torch.as_tensor(filtered_orders["o_custkey"].to_cupy(), dtype=torch.int64).cuda()

    # Get max customer key for histogram size
    num_customers = int(df_customer["c_custkey"].max())

    # Run optimized CUDA pipeline
    c_count_t, custdist_t = _query_core(c_custkey_t, o_custkey_t, num_customers)

    return cudf.DataFrame({
        "c_count": c_count_t,
        "custdist": custdist_t,
    })
