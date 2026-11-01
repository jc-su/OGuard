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

#define BLOCK_SIZE 256

__global__ void q18_threshold_kernel(
    const int64_t* __restrict__ sorted_keys,
    const double* __restrict__ sorted_qty,
    int64_t* __restrict__ out_keys,
    double* __restrict__ out_sums,
    int64_t* __restrict__ out_count,
    double qty_threshold,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    bool is_start = (idx == 0) || (sorted_keys[idx] != sorted_keys[idx - 1]);
    if (!is_start) return;
    
    double sum = 0.0;
    int64_t key = sorted_keys[idx];
    int64_t j = idx;
    while (j < n && sorted_keys[j] == key) {
        sum += sorted_qty[j];
        j++;
    }
    
    if (sum > qty_threshold) {
        int64_t pos = atomicAdd((unsigned long long int*)out_count, 1ULL);
        out_keys[pos] = key;
        out_sums[pos] = sum;
    }
}

std::vector<torch::Tensor> q18_launcher(
    torch::Tensor orderkey,
    torch::Tensor quantity,
    double qty_threshold
) {
    TORCH_CHECK(orderkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64);
    
    auto orderkey_c = orderkey.contiguous().cuda();
    auto quantity_c = quantity.contiguous().cuda();
    int64_t n = orderkey_c.numel();
    
    // Sort by orderkey to group same keys
    auto sorted_idx = std::get<1>(torch::sort(orderkey_c, 0, false));
    auto sorted_keys = orderkey_c.index_select(0, sorted_idx).contiguous();
    auto sorted_qty = quantity_c.index_select(0, sorted_idx).contiguous();
    
    auto out_keys = torch::empty({n}, orderkey_c.options());
    auto out_sums = torch::empty({n}, quantity_c.options());
    auto out_count = torch::zeros({1}, torch::kInt64).cuda();
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    q18_threshold_kernel<<<grid, block>>>(
        sorted_keys.data_ptr<int64_t>(),
        sorted_qty.data_ptr<double>(),
        out_keys.data_ptr<int64_t>(),
        out_sums.data_ptr<double>(),
        out_count.data_ptr<int64_t>(),
        qty_threshold,
        n
    );
    
    int64_t h_count = out_count.cpu().item<int64_t>();
    if (h_count > 0) {
        return {out_keys.slice(0, 0, h_count).clone(), out_sums.slice(0, 0, h_count).clone()};
    }
    return {torch::empty({0}, orderkey_c.options()), torch::empty({0}, quantity_c.options())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q18_launcher", &q18_launcher, "TPC-H Q18 CUDA launcher");
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
        extra_cuda_cflags=["-O3", "-arch=sm_90", "--use_fast_math"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    qty_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA-accelerated group-by sum with threshold filter."""
    big_keys, big_sums = _EXT.q18_launcher(l_orderkey, l_quantity, qty_threshold)
    sorted_all = torch.sort(torch.unique(l_orderkey))[0]
    return big_keys, big_sums, sorted_all

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

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    big_keys_t, big_sums_t, _ = _query_core(l_orderkey_t, l_quantity_t, float(qty_threshold))

    df_big = cudf.DataFrame({
        "o_orderkey": cudf.Series(big_keys_t),
        "sum_qty": cudf.Series(big_sums_t),
    })

    orders_filtered = df_orders.merge(df_big, left_on="o_orderkey", right_on="o_orderkey", how="inner")
    joined = orders_filtered.merge(
        df_customer[["c_custkey", "c_name"]],
        left_on="o_custkey", right_on="c_custkey", how="inner"
    )

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
