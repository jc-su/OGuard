import os
import tempfile
from typing import Dict, Tuple

import cudf
import torch
from torch.utils.cpp_extension import load


# -------------------------------------------------------------------------
# CUDA source: sort+reduce based group aggregation for lineitem quantities.
# -------------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>

// Kernel that filters groups whose sum exceeds threshold.
// Writes results into pre‑allocated buffers using an atomic counter.
__global__ void filter_big_groups(
    const int64_t* __restrict__ keys,
    const double*   __restrict__ sums,
    int64_t* __restrict__ out_keys,
    double*   __restrict__ out_sums,
    double threshold,
    int64_t n,
    unsigned long long* __restrict__ count_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (sums[idx] > threshold) {
        unsigned long long pos = atomicAdd(count_out, 1ULL);
        out_keys[pos] = keys[idx];
        out_sums[pos] = sums[idx];
    }
}

// Main launcher: returns (big_keys, big_sums, unique_keys) as torch tensors.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> q18_launcher(
    torch::Tensor orderkeys,
    torch::Tensor quantities,
    double threshold)
{
    TORCH_CHECK(orderkeys.is_cuda(), "orderkeys must be a CUDA tensor");
    TORCH_CHECK(quantities.is_cuda(), "quantities must be a CUDA tensor");
    TORCH_CHECK(orderkeys.scalar_type() == torch::kInt64,   "orderkeys must be int64");
    TORCH_CHECK(quantities.scalar_type() == torch::kFloat64, "quantities must be float64");

    // Ensure contiguous GPU memory
    orderkeys = orderkeys.contiguous();
    quantities = quantities.contiguous();

    int64_t n = orderkeys.size(0);

    // Wrap raw device pointers for Thrust
    thrust::device_ptr<int64_t> keys_ptr =
        thrust::device_pointer_cast(orderkeys.data_ptr<int64_t>());
    thrust::device_ptr<double> qty_ptr =
        thrust::device_pointer_cast(quantities.data_ptr<double>());

    // -----------------------------------------------------------------
    // 1) Sort by orderkey (ascending). This also orders quantities accordingly.
    // -----------------------------------------------------------------
    thrust::sort_by_key(keys_ptr, keys_ptr + n, qty_ptr);

    // -----------------------------------------------------------------
    // 2) Reduce by key – compute sum of quantities for each unique orderkey.
    // -----------------------------------------------------------------
    thrust::device_vector<int64_t> unique_keys_vec(n);
    thrust::device_vector<double>   unique_sums_vec(n);

    auto new_end = thrust::reduce_by_key(
        keys_ptr, keys_ptr + n,
        qty_ptr,
        unique_keys_vec.begin(),
        unique_sums_vec.begin(),
        thrust::equal_to<int64_t>(),
        thrust::plus<double>());

    size_t unique_count = thrust::distance(unique_keys_vec.begin(), new_end.first);
    unique_keys_vec.resize(unique_count);
    unique_sums_vec.resize(unique_count);

    // -----------------------------------------------------------------
    // 3) Filter groups whose sum > threshold.
    // -----------------------------------------------------------------
    // Allocate buffers of maximal possible size (unique_count).
    thrust::device_vector<int64_t> big_keys_vec(unique_count);
    thrust::device_vector<double>  big_sums_vec(unique_count);
    thrust::device_vector<unsigned long long> big_count_vec(1);
    big_count_vec[0] = 0;
    unsigned long long* big_count_ptr = big_count_vec.data().get();

    const int64_t BLOCK_SIZE = 256;
    int64_t grid = (unique_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    filter_big_groups<<<grid, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(unique_keys_vec.data()),
        thrust::raw_pointer_cast(unique_sums_vec.data()),
        thrust::raw_pointer_cast(big_keys_vec.data()),
        thrust::raw_pointer_cast(big_sums_vec.data()),
        threshold,
        unique_count,
        big_count_ptr);
    cudaDeviceSynchronize();

    // Copy count back to host to know actual size of result.
    unsigned long long big_count = 0;
    cudaMemcpy(&big_count, big_count_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    big_keys_vec.resize(big_count);
    big_sums_vec.resize(big_count);

    // -----------------------------------------------------------------
    // 4) Convert results to torch::Tensor and return.
    // -----------------------------------------------------------------
    // Unique keys (all distinct orderkeys)
    torch::Tensor unique_out = torch::empty({static_cast<long>(unique_count)},
                                            orderkeys.options());
    thrust::copy(unique_keys_vec.begin(), unique_keys_vec.end(),
                 unique_out.data_ptr<int64_t>());

    // Big keys
    torch::Tensor big_keys_out = torch::empty({static_cast<long>(big_count)}, orderkeys.options());
    if (big_count > 0) {
        thrust::copy(big_keys_vec.begin(), big_keys_vec.end(),
                     big_keys_out.data_ptr<int64_t>());
    }

    // Big sums
    torch::Tensor big_sums_out = torch::empty({static_cast<long>(big_count)}, quantities.options());
    if (big_count > 0) {
        thrust::copy(big_sums_vec.begin(), big_sums_vec.end(),
                     big_sums_out.data_ptr<double>());
    }

    return std::make_tuple(big_keys_out, big_sums_out, unique_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q18_launcher", &q18_launcher,
          "Compute per‑order quantity sums and filter > threshold (CUDA)");
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


# Load the compiled extension once at import time.
_EXT = _load_cuda()


# -------------------------------------------------------------------------
# Core GPU routine: group‑by orderkey, sum quantity, filter > threshold.
# -------------------------------------------------------------------------
def _query_core(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    qty_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes per‑order sum of l_quantity, filters orders where the sum exceeds
    qty_threshold, and returns:
        - big_keys: orderkeys with sum > threshold (sorted ascending)
        - big_sums : corresponding sums (float64)
        - unique_keys: all distinct orderkeys (sorted ascending)
    The implementation uses a fused CUDA pipeline:
        1) Sort by orderkey
        2) Thrust reduce‑by‑key to compute per‑order sums
        3) Filter groups > threshold with a custom kernel
    This avoids the multiple passes (unique + scatter) of the reference and
    keeps the heavy aggregation entirely on the GPU.
    """
    # Ensure tensors are on GPU and contiguous
    l_orderkey = l_orderkey.contiguous()
    l_quantity = l_quantity.contiguous()

    # Call the compiled CUDA launcher
    big_keys, big_sums, unique_keys = _EXT.q18_launcher(
        l_orderkey, l_quantity, float(qty_threshold)
    )
    return big_keys, big_sums, unique_keys


# -------------------------------------------------------------------------
# Main query: TPC‑H Q18 – largest‑quantity orders.
# -------------------------------------------------------------------------
def run_query(
    tables: Dict[str, cudf.DataFrame],
    qty_threshold: float = 300.0,
    limit: int = 100,
) -> cudf.DataFrame:
    """
    SQL:
      SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice,
             sum(l_quantity)
      FROM customer, orders, lineitem
      WHERE o_orderkey IN (
            SELECT l_orderkey FROM lineitem
            GROUP BY l_orderkey HAVING sum(l_quantity) > 300)
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
      GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
      ORDER BY o_totalprice DESC, o_orderdate
      LIMIT 100;

    Description:
      This implements TPC‑H Q18 using a CUDA‑accelerated group‑by and filter
      on the lineitem table. The per‑order quantity sums are computed on the
      GPU with a single sort + reduce pass, then filtered (> qty_threshold).
      The remaining joins and final sorting are performed by cuDF, which is
      efficient for the smaller dimension tables (orders, customer).

    Params:
        tables: Dictionary mapping table name to cudf.DataFrame.
        qty_threshold: HAVING sum(l_quantity) > threshold (default 300.0).
        limit: Number of rows to return (default 100).
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # --------------------------------------------------------------
    # 1) GPU‑accelerated aggregation on lineitem (hot path)
    # --------------------------------------------------------------
    # Convert cuDF columns to CUDA tensors using torch.as_tensor.
    l_orderkey_t = torch.as_tensor(
        df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64, device="cuda"
    )
    l_quantity_t = torch.as_tensor(
        df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64, device="cuda"
    )

    big_keys_t, big_sums_t, _ = _query_core(
        l_orderkey_t, l_quantity_t, float(qty_threshold)
    )

    # --------------------------------------------------------------
    # 2) Build a small cuDF table with qualifying orderkeys + sums
    # --------------------------------------------------------------
    df_big = cudf.DataFrame({
        "o_orderkey": cudf.Series(big_keys_t),
        "sum_qty": cudf.Series(big_sums_t),
    })

    # --------------------------------------------------------------
    # 3) Semi‑join orders with the big orderkeys, then join customer
    # --------------------------------------------------------------
    orders_filtered = df_orders.merge(
        df_big, left_on="o_orderkey", right_on="o_orderkey", how="inner"
    )
    joined = orders_filtered.merge(
        df_customer[["c_custkey", "c_name"]],
        left_on="o_custkey", right_on="c_custkey", how="inner"
    )

    # --------------------------------------------------------------
    # 4) Sort and limit
    # --------------------------------------------------------------
    ordered = joined.sort_values(
        by=["o_totalprice", "o_orderdate"], ascending=[False, True]
    ).head(limit)

    # --------------------------------------------------------------
    # 5) Project final columns in required order
    # --------------------------------------------------------------
    result = cudf.DataFrame({
        "c_name": ordered["c_name"].reset_index(drop=True),
        "c_custkey": ordered["c_custkey"].reset_index(drop=True),
        "o_orderkey": ordered["o_orderkey"].reset_index(drop=True),
        "o_orderdate": ordered["o_orderdate"].reset_index(drop=True),
        "o_totalprice": ordered["o_totalprice"].reset_index(drop=True),
        "sum(l_quantity)": ordered["sum_qty"].reset_index(drop=True),
    })
    return result
