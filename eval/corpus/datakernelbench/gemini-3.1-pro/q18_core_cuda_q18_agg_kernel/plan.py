import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q18_agg_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ values,
    double* __restrict__ sums,
    uint8_t* __restrict__ exists,
    int64_t n)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t k = keys[i];
        double v = values[i];
        
        // Fast hardware-accelerated L2 atomic additions on Hopper architectures
        atomicAdd(&sums[k], v);
        exists[k] = 1;
    }
}

void q18_launcher(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor sums,
    torch::Tensor exists)
{
    TORCH_CHECK(keys.scalar_type() == torch::kInt64);
    TORCH_CHECK(values.scalar_type() == torch::kFloat64);
    TORCH_CHECK(sums.scalar_type() == torch::kFloat64);
    TORCH_CHECK(exists.scalar_type() == torch::kUInt8);
    
    int64_t n = keys.numel();
    if (n == 0) return;
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q18_agg_kernel<<<blocks, BLOCK_SIZE>>>(
        keys.data_ptr<int64_t>(),
        values.data_ptr<double>(),
        sums.data_ptr<double>(),
        exists.data_ptr<uint8_t>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q18_launcher", &q18_launcher, "TPC-H Q18 aggregation");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q18_cuda_")
    cu_path = os.path.join(tmpdir, "_q18_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q18_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    qty_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replaces the slow torch.unique + scatter_add with a dense CUDA atomic reduction.
    Requires keys to be positive integers (true for TPC-H orderkeys).
    Provides enormous speedups by avoiding explicit device-side sorting and hash tables.
    """
    max_val = int(l_orderkey.max().item()) if l_orderkey.numel() > 0 else 0
    M = max_val + 1
    
    sums = torch.zeros(M, dtype=torch.float64, device=l_orderkey.device)
    exists = torch.zeros(M, dtype=torch.uint8, device=l_orderkey.device)
    
    if l_orderkey.numel() > 0:
        _EXT.q18_launcher(
            l_orderkey.contiguous(),
            l_quantity.contiguous(),
            sums,
            exists
        )
    
    # torch.nonzero naturally returns elements ordered by index, matching torch.unique
    unique_keys = torch.nonzero(exists, as_tuple=True)[0]
    all_sums = sums[unique_keys]
    
    big_mask = all_sums > qty_threshold
    big_keys = unique_keys[big_mask]
    big_sums = all_sums[big_mask]
    
    return big_keys, big_sums, unique_keys

def run_query(tables: dict[str, cudf.DataFrame], qty_threshold: float = 300.0, limit: int = 100) -> cudf.DataFrame:
    """
    SQL:
      SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
      FROM customer, orders, lineitem
      WHERE o_orderkey IN (
          SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300)
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
      GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
      ORDER BY o_totalprice DESC, o_orderdate
      LIMIT 100;
    Description: TPC-H Q18 — finds the top 100 customers with the largest-quantity orders.
    Params:
      qty_threshold (float): HAVING sum(l_quantity) > threshold.
      limit (int): number of rows to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # --- Compute big orderkeys (HAVING sum(l_quantity) > threshold) in the core ---
    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    big_keys_t, big_sums_t, _ = _query_core(l_orderkey_t, l_quantity_t, float(qty_threshold))

    # Build a cuDF DataFrame of qualifying orderkeys + their total quantity
    df_big = cudf.DataFrame({
        "o_orderkey": cudf.Series(big_keys_t),
        "sum_qty": cudf.Series(big_sums_t),
    })

    # Semi-join orders with big orderkeys, then join with customer
    orders_filtered = df_orders.merge(df_big, left_on="o_orderkey", right_on="o_orderkey", how="inner")
    joined = orders_filtered.merge(
        df_customer[["c_custkey", "c_name"]],
        left_on="o_custkey", right_on="c_custkey", how="inner"
    )

    # Order by o_totalprice DESC, o_orderdate ASC
    joined = joined.sort_values(
        by=["o_totalprice", "o_orderdate"], ascending=[False, True]
    ).head(limit)

    result = cudf.DataFrame({
        "c_name": joined["c_name"].reset_index(drop=True),
        "c_custkey": joined["c_custkey"].reset_index(drop=True),
        "o_orderkey": joined["o_orderkey"].reset_index(drop=True),
        "o_orderdate": joined["o_orderdate"].reset_index(drop=True),
        "o_totalprice": joined["o_totalprice"].reset_index(drop=True),
        "sum(l_quantity)": joined["sum_qty"].reset_index(drop=True),
    })
    return result
