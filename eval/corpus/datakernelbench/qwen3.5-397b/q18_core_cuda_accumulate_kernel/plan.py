import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 1024

__global__ void accumulate_kernel(
    const int64_t* __restrict__ orderkeys,
    const double* __restrict__ quantities,
    double* __restrict__ sums,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = orderkeys[idx];
        double qty = quantities[idx];
        // Direct addressing: key is 1-based index into sums
        // Ensure key is within bounds (should be, based on TPC-H spec)
        if (key > 0) {
            atomicAdd(&sums[key], qty);
        }
    }
}

__global__ void extract_kernel(
    const double* __restrict__ sums,
    int64_t* __restrict__ out_all_keys,
    int64_t* __restrict__ out_big_keys,
    double* __restrict__ out_big_sums,
    int32_t* __restrict__ count_all,
    int32_t* __restrict__ count_big,
    int64_t num_slots,
    double threshold
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_slots) {
        double s = sums[idx];
        if (s > 0.0) {
            int32_t pos = atomicAdd(count_all, 1);
            out_all_keys[pos] = idx;
        }
        if (s > threshold) {
            int32_t pos = atomicAdd(count_big, 1);
            out_big_keys[pos] = idx;
            out_big_sums[pos] = s;
        }
    }
}

std::vector<torch::Tensor> q18_launcher(
    torch::Tensor orderkeys,
    torch::Tensor quantities,
    double qty_threshold
) {
    // Ensure inputs are on CUDA and contiguous
    if (!orderkeys.is_cuda()) orderkeys = orderkeys.cuda();
    if (!quantities.is_cuda()) quantities = quantities.cuda();
    orderkeys = orderkeys.contiguous();
    quantities = quantities.contiguous();

    TORCH_CHECK(orderkeys.scalar_type() == torch::kInt64, "orderkeys must be int64");
    TORCH_CHECK(quantities.scalar_type() == torch::kFloat64, "quantities must be float64");

    int64_t n = orderkeys.numel();
    if (n == 0) {
        return {
            torch::empty({0}, torch::kInt64),
            torch::empty({0}, torch::kFloat64),
            torch::empty({0}, torch::kInt64)
        };
    }

    // Find max key to allocate sums array
    // Using torch::max which returns a tensor
    torch::Tensor max_key_tensor = torch::max(orderkeys);
    int64_t max_key = max_key_tensor.item<int64_t>();
    
    // TPC-H keys are 1-based. We need size max_key + 1 to index directly.
    int64_t num_slots = max_key + 1;

    // Allocate sums array on device
    auto options = torch::dtype(torch::kFloat64).device(orderkeys.device());
    torch::Tensor sums = torch::zeros({num_slots}, options);

    // Launch accumulation kernel
    int blocks_acc = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    accumulate_kernel<<<blocks_acc, BLOCK_SIZE>>>(
        orderkeys.data_ptr<int64_t>(),
        quantities.data_ptr<double>(),
        sums.data_ptr<double>(),
        n
    );
    cudaDeviceSynchronize();

    // Prepare output buffers (max possible size = num_slots)
    auto key_options = torch::dtype(torch::kInt64).device(orderkeys.device());
    torch::Tensor out_all_keys = torch::empty({num_slots}, key_options);
    torch::Tensor out_big_keys = torch::empty({num_slots}, key_options);
    torch::Tensor out_big_sums = torch::empty({num_slots}, options);

    // Counters for output sizes
    torch::Tensor count_all = torch::zeros({1}, torch::dtype(torch::kInt32).device(orderkeys.device()));
    torch::Tensor count_big = torch::zeros({1}, torch::dtype(torch::kInt32).device(orderkeys.device()));

    // Launch extract kernel
    int blocks_ext = (num_slots + BLOCK_SIZE - 1) / BLOCK_SIZE;
    extract_kernel<<<blocks_ext, BLOCK_SIZE>>>(
        sums.data_ptr<double>(),
        out_all_keys.data_ptr<int64_t>(),
        out_big_keys.data_ptr<int64_t>(),
        out_big_sums.data_ptr<double>(),
        count_all.data_ptr<int32_t>(),
        count_big.data_ptr<int32_t>(),
        num_slots,
        qty_threshold
    );
    cudaDeviceSynchronize();

    // Get counts
    int32_t c_all = count_all.item<int32_t>();
    int32_t c_big = count_big.item<int32_t>();

    // Slice tensors to actual size
    // narrow(dim, start, length)
    torch::Tensor final_all_keys = out_all_keys.narrow(0, 0, c_all);
    torch::Tensor final_big_keys = out_big_keys.narrow(0, 0, c_big);
    torch::Tensor final_big_sums = out_big_sums.narrow(0, 0, c_big);

    return {final_big_keys, final_big_sums, final_all_keys};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q18_launcher", &q18_launcher, "TPC-H Q18 Core Aggregation (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q18_cuda_")
    cu_path = os.path.join(tmpdir, "_q18_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    
    # H100 is sm_90
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
    Computes per-order sum of l_quantity using a direct-addressing CUDA kernel,
    then filters those orders whose summed quantity exceeds the threshold.
    Returns the unique big order keys, along with (for the eventual join) 
    sorted unique orderkeys + sums.
    
    Optimization: Replaces torch.unique + scatter_add with a single-pass 
    direct-addressing accumulation (O(N) vs O(N log N)) since TPC-H orderkeys 
    are dense integers.
    """
    # Ensure tensors are contiguous and on CUDA for the extension
    if not l_orderkey.is_cuda:
        l_orderkey = l_orderkey.cuda()
    if not l_quantity.is_cuda:
        l_quantity = l_quantity.cuda()
    
    # Call CUDA extension
    # Returns: big_keys, big_sums, unique_keys
    res = _EXT.q18_launcher(l_orderkey, l_quantity, float(qty_threshold))
    return res[0], res[1], res[2]

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
