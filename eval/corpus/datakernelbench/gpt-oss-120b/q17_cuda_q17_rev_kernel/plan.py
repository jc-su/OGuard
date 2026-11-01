import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source – all device code lives in this string and is compiled at import.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cstdint>

#define BLOCK_SIZE 1024

// binary search for an int64 key in a sorted int64 array.
// Returns index if found, otherwise -1.
__device__ int64_t binary_search(const int64_t* arr, int64_t n, int64_t key) {
    int64_t lo = 0;
    int64_t hi = n - 1;
    while (lo <= hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t val = arr[mid];
        if (val == key) {
            return mid;
        } else if (val < key) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

/* --------------------------------------------------------------------
   First kernel: for every lineitem row that belongs to a filtered part
   (i.e. its partkey appears in `sorted_keys`), add its quantity to a
   per‑part sum and increment a per‑part count.
   -------------------------------------------------------------------- */
__global__ void q17_sumcnt_kernel(
    const int64_t* __restrict__ line_partkey,
    const double* __restrict__ line_quantity,
    const int64_t* __restrict__ sorted_keys,
    int64_t n_parts,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n_rows)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    int64_t key = line_partkey[idx];
    int64_t part_idx = binary_search(sorted_keys, n_parts, key);
    if (part_idx >= 0) {
        // atomically accumulate quantity and count (as double to stay fp64)
        atomicAdd(&sum_qty[part_idx], line_quantity[idx]);
        atomicAdd(&cnt_qty[part_idx], 1.0);
    }
}

/* --------------------------------------------------------------------
   Second kernel: compute revenue = sum(l_extendedprice) over rows where
   l_quantity < 0.2 * avg_quantity_for_part.
   Thresholds are pre‑computed on the host and passed in `thresholds`.
   -------------------------------------------------------------------- */
__global__ void q17_rev_kernel(
    const int64_t* __restrict__ line_partkey,
    const double* __restrict__ line_quantity,
    const double* __restrict__ line_price,
    const int64_t* __restrict__ sorted_keys,
    const double* __restrict__ thresholds,
    int64_t n_parts,
    double* __restrict__ out_sum,
    int64_t n_rows)
{
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;

    if (idx < n_rows) {
        int64_t key = line_partkey[idx];
        int64_t part_idx = binary_search(sorted_keys, n_parts, key);
        if (part_idx >= 0) {
            double qty = line_quantity[idx];
            double th = thresholds[part_idx];
            if (qty < th) {
                my_rev = line_price[idx];
            }
        }
    }
    block_sum[threadIdx.x] = my_rev;
    __syncthreads();

    // intra‑block reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    // one atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, block_sum[0]);
    }
}

/* --------------------------------------------------------------------
   Launcher for the first kernel – populates sum_qty and cnt_qty.
   -------------------------------------------------------------------- */
void q17_sumcnt_launcher(
    torch::Tensor line_partkey,
    torch::Tensor line_quantity,
    torch::Tensor sorted_keys,
    torch::Tensor sum_qty,
    torch::Tensor cnt_qty)
{
    TORCH_CHECK(line_partkey.is_cuda() && line_quantity.is_cuda() && sorted_keys.is_cuda(),
                "All tensors must be on CUDA");
    TORCH_CHECK(line_partkey.dtype() == torch::kInt64, "line_partkey must be int64");
    TORCH_CHECK(line_quantity.dtype() == torch::kFloat64, "line_quantity must be float64");
    TORCH_CHECK(sorted_keys.dtype() == torch::kInt64, "sorted_keys must be int64");
    TORCH_CHECK(sum_qty.dtype() == torch::kFloat64 && cnt_qty.dtype() == torch::kFloat64,
                "sum/cnt must be float64");

    const int64_t n_rows = line_partkey.numel();
    const int64_t n_parts = sorted_keys.numel();

    const int threads = BLOCK_SIZE;
    const int blocks = (n_rows + threads - 1) / threads;

    q17_sumcnt_kernel<<<blocks, threads>>>(
        line_partkey.data_ptr<int64_t>(),
        line_quantity.data_ptr<double>(),
        sorted_keys.data_ptr<int64_t>(),
        n_parts,
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        n_rows);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

/* --------------------------------------------------------------------
   Launcher for the second kernel – returns the total revenue as double.
   -------------------------------------------------------------------- */
double q17_rev_launcher(
    torch::Tensor line_partkey,
    torch::Tensor line_quantity,
    torch::Tensor line_price,
    torch::Tensor sorted_keys,
    torch::Tensor thresholds)
{
    TORCH_CHECK(line_partkey.is_cuda() && line_quantity.is_cuda() && line_price.is_cuda()
                && sorted_keys.is_cuda() && thresholds.is_cuda(),
                "All tensors must be on CUDA");
    TORCH_CHECK(line_partkey.dtype() == torch::kInt64, "line_partkey must be int64");
    TORCH_CHECK(line_quantity.dtype() == torch::kFloat64, "line_quantity must be float64");
    TORCH_CHECK(line_price.dtype() == torch::kFloat64, "line_price must be float64");
    TORCH_CHECK(sorted_keys.dtype() == torch::kInt64, "sorted_keys must be int64");
    TORCH_CHECK(thresholds.dtype() == torch::kFloat64, "thresholds must be float64");

    const int64_t n_rows = line_partkey.numel();
    const int64_t n_parts = sorted_keys.numel();

    // output buffer on device (single double)
    auto out = torch::zeros({1}, line_price.options());  // float64, CUDA
    const int threads = BLOCK_SIZE;
    const int blocks = (n_rows + threads - 1) / threads;

    q17_rev_kernel<<<blocks, threads>>>(
        line_partkey.data_ptr<int64_t>(),
        line_quantity.data_ptr<double>(),
        line_price.data_ptr<double>(),
        sorted_keys.data_ptr<int64_t>(),
        thresholds.data_ptr<double>(),
        n_parts,
        out.data_ptr<double>(),
        n_rows);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
    // Copy result back to host and return
    double host_res = out.item<double>();
    return host_res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_sumcnt_launcher", &q17_sumcnt_launcher,
          "Launch kernel that computes per‑part sum and count of quantity");
    m.def("q17_rev_launcher", &q17_rev_launcher,
          "Launch kernel that computes revenue given per‑part thresholds");
}
'''

# ----------------------------------------------------------------------
# Helper to compile and load the CUDA extension (once per import).
# ----------------------------------------------------------------------
def _load_cuda():
    """Compile the in‑memory CUDA source and return the loaded module."""
    tmpdir = tempfile.mkdtemp(prefix="q17_cuda_")
    src_path = os.path.join(tmpdir, "q17_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(src_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q17_cuda_ext",
        sources=[src_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

_EXT = _load_cuda()  # compiled extension

# ----------------------------------------------------------------------
# Core query logic – uses the compiled kernels.
# ----------------------------------------------------------------------
def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC‑H Q17 using two fused CUDA kernels:
      1. Accumulate per‑part sum(quantity) and count.
      2. Compute revenue where quantity < 0.2 * avg_quantity.
    The result is the total revenue divided by 7.0.
    """
    device = l_partkey.device

    # 1️⃣ Sort the filtered part keys (GPU sort, cheap)
    sorted_keys, _ = torch.sort(part_keys)

    n_parts = sorted_keys.numel()
    # Allocate per‑part aggregates on device, zero‑initialized
    sum_qty = torch.zeros(n_parts, dtype=torch.float64, device=device)
    cnt_qty = torch.zeros(n_parts, dtype=torch.float64, device=device)

    # Launch kernel that fills sum_qty and cnt_qty
    _EXT.q17_sumcnt_launcher(
        l_partkey, l_quantity, sorted_keys, sum_qty, cnt_qty
    )

    # Compute per‑part average quantity and 0.2 * avg (still on GPU)
    avg_qty = torch.where(cnt_qty > 0.0, sum_qty / cnt_qty,
                          torch.zeros_like(sum_qty))
    thresholds = 0.2 * avg_qty

    # 2️⃣ Launch kernel that aggregates revenue using the thresholds
    revenue = _EXT.q17_rev_launcher(
        l_partkey, l_quantity, l_extendedprice, sorted_keys, thresholds
    )
    # revenue is a Python float (copied from device); convert to tensor for consistency
    rev_tensor = torch.tensor(revenue, dtype=torch.float64, device=device)
    return rev_tensor / 7.0


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (
            SELECT 0.2 * avg(l_quantity)
            FROM lineitem
            WHERE l_partkey = p_partkey);
    Description:
        This implementation keeps the join/filter on the GPU.  The `part` table is
        filtered with cuDF (GPU‑accelerated), then the list of qualifying partkeys
        is transferred to a Torch tensor.  Two custom CUDA kernels perform the
        expensive grouped aggregation in a fused, memory‑bound manner:
          • First kernel computes per‑part sum and count of `l_quantity`.
          • Host converts these to the 0.2 × average thresholds.
          • Second kernel computes the revenue sum in a single pass using
            block‑level reductions and a single atomic per block.
        By fusing the scan and performing all reductions on‑device we avoid the
        multiple cuDF → Torch round‑trips of the reference plan, reducing total
        memory traffic and achieving a faster overall runtime on H100.
    Params:
        brand_val (str): part brand filter literal.
        container_val (str): part container filter literal.
    Returns:
        float: average yearly revenue (sum(l_extendedprice)/7.0).
    """
    # ------------------------------------------------------------------
    # 1️⃣ Part table: filter by brand and container using cuDF (GPU resident)
    # ------------------------------------------------------------------
    df_part = tables["part"]
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    # ------------------------------------------------------------------
    # 2️⃣ Convert needed columns to Torch tensors (already on GPU via cupy)
    # ------------------------------------------------------------------
    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)

    df_lineitem = tables["lineitem"]
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    # ------------------------------------------------------------------
    # 3️⃣ Run the fused GPU kernels
    # ------------------------------------------------------------------
    result_tensor = _query_core(
        l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t
    )
    return float(result_tensor.item())
