import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 1024

__global__ void group_sum_kernel(
    const double* __restrict__ qty,
    const int64_t* __restrict__ group_idx,
    double* __restrict__ group_sum,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t g = group_idx[idx];
        atomicAdd(&group_sum[g], qty[idx]);
    }
}

torch::Tensor group_sum_launcher(
    torch::Tensor qty,
    torch::Tensor group_idx,
    int64_t num_groups
) {
    TORCH_CHECK(qty.is_cuda(), "qty must be a CUDA tensor");
    TORCH_CHECK(group_idx.is_cuda(), "group_idx must be a CUDA tensor");
    TORCH_CHECK(qty.scalar_type() == torch::kFloat64, "qty must be float64");
    TORCH_CHECK(group_idx.scalar_type() == torch::kInt64, "group_idx must be int64");

    qty = qty.contiguous();
    group_idx = group_idx.contiguous();

    auto group_sum = torch::zeros({num_groups}, qty.options());

    int64_t n = qty.numel();
    int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    group_sum_kernel<<<blocks, BLOCK_SIZE>>>(
        qty.data_ptr<double>(),
        group_idx.data_ptr<int64_t>(),
        group_sum.data_ptr<double>(),
        n
    );

    cudaDeviceSynchronize();
    return group_sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_sum_launcher", &group_sum_launcher, "Group sum via CUDA");
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
    Computes per-order sum of l_quantity, then filters those orders whose
    summed quantity exceeds the threshold. Returns the unique big order keys,
    along with (for the eventual join) sorted unique orderkeys + sums.
    This implementation uses a custom CUDA kernel for the group accumulation,
    while the sorting and filtering steps leverage native PyTorch (CUDA) ops.
    """
    # Ensure inputs are on CUDA and contiguous
    if not l_orderkey.is_cuda:
        l_orderkey = l_orderkey.cuda()
    if not l_quantity.is_cuda:
        l_quantity = l_quantity.cuda()
    l_orderkey = l_orderkey.contiguous()
    l_quantity = l_quantity.contiguous()

    # 1) Sort order keys; obtain indices to reorder quantities accordingly
    sorted_keys, indices = torch.sort(l_orderkey)
    sorted_qty = l_quantity[indices]

    # 2) Identify group boundaries (start of each new order key)
    n = sorted_keys.shape[0]
    boundaries = torch.empty(n, dtype=torch.bool, device=sorted_keys.device)
    boundaries[0] = True
    if n > 1:
        boundaries[1:] = sorted_keys[1:] != sorted_keys[:-1]

    # 3) Compute group index for each element (0‑based group id)
    group_idx = torch.cumsum(boundaries.to(torch.int64), dim=0) - 1

    # 4) Number of distinct order keys
    num_groups = int(boundaries.sum().item())

    # 5) Compute per‑group sum using custom CUDA kernel
    group_sum = _EXT.group_sum_launcher(sorted_qty, group_idx, num_groups)

    # Ensure the kernel has finished before we use group_sum
    torch.cuda.synchronize()

    # 6) Extract the sorted unique order keys (the first element of each group)
    unique_keys = sorted_keys[boundaries]

    # 7) Filter groups where sum > qty_threshold
    mask = group_sum > qty_threshold
    big_keys = unique_keys[mask]
    big_sums = group_sum[mask]

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
