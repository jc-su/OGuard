import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q13_hash_kernel(
    const int64_t* __restrict__ c_custkey,
    int64_t num_customers,
    const int64_t* __restrict__ o_custkey,
    int64_t n_orders,
    unsigned long long* __restrict__ order_counts
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_orders) {
        int64_t cust = o_custkey[tid];
        if (cust >= 1 && cust <= num_customers) {
            atomicAdd(&order_counts[cust], 1ULL);
        }
    }
}

__global__ void q13_gather_kernel(
    const int64_t* __restrict__ c_custkey,
    const unsigned long long* __restrict__ order_counts,
    int64_t n_customers,
    int64_t* __restrict__ c_count
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_customers) {
        int64_t cust = c_custkey[tid];
        c_count[tid] = (int64_t)order_counts[cust];
    }
}

torch::Tensor do_hash_launcher(
    torch::Tensor c_custkey,
    torch::Tensor o_custkey,
    int64_t num_customers,
    torch::Tensor order_counts
) {
    TORCH_CHECK(c_custkey.is_cuda());
    TORCH_CHECK(o_custkey.is_cuda());
    TORCH_CHECK(order_counts.is_cuda());
    
    int64_t n_orders = o_custkey.numel();
    int64_t n_cust = c_custkey.numel();
    
    int blocks = (n_orders + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q13_hash_kernel<<<blocks, BLOCK_SIZE>>>(
        c_custkey.data_ptr<int64_t>(),
        num_customers,
        o_custkey.data_ptr<int64_t>(),
        n_orders,
        (unsigned long long*)order_counts.data_ptr<int64_t>()
    );
    
    int gather_blocks = (n_cust + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto c_count = torch::zeros({n_cust}, c_custkey.options());
    q13_gather_kernel<<<gather_blocks, BLOCK_SIZE>>>(
        c_custkey.data_ptr<int64_t>(),
        (const unsigned long long*)order_counts.data_ptr<int64_t>(),
        n_cust,
        c_count.data_ptr<int64_t>()
    );
    
    return c_count;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("do_hash", &do_hash_launcher, "Compute order counts per customer and gather");
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

def _launch_q13(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> torch.Tensor:
    """
    Launch CUDA kernels to compute order counts per customer.
    Returns c_count tensor (per-customer order count).
    """
    c_custkey = c_custkey.contiguous()
    o_custkey_filtered = o_custkey_filtered.contiguous()
    
    order_counts = torch.zeros(num_customers + 1, dtype=torch.int64, device=c_custkey.device)
    
    c_count = _EXT.do_hash(c_custkey, o_custkey_filtered, num_customers, order_counts)
    return c_count

def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute customer distribution (Q13 core logic) using fused CUDA kernel.
    
    GPU: fused hash + gather on CUDA, group-by and sort on GPU via torch.
    Returns (unique_counts, custdist) following reference semantics.
    """
    c_count = _launch_q13(c_custkey, o_custkey_filtered, num_customers)
    
    unique_counts, custdist = torch.unique(c_count, return_counts=True)
    
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

    mask = ~df_orders["o_comment"].str.contains("special.*requests", regex=True)
    filtered_orders = df_orders[mask]

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    o_custkey_t = torch.as_tensor(filtered_orders["o_custkey"].to_cupy(), dtype=torch.int64)

    num_customers = int(df_customer["c_custkey"].max())

    c_count_t, custdist_t = _query_core(c_custkey_t, o_custkey_t, num_customers)

    return cudf.DataFrame({
        "c_count": c_count_t.cpu(),
        "custdist": custdist_t.cpu(),
    })
