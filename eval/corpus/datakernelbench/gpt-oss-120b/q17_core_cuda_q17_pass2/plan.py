import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: two fused kernels + launcher for TPC-H Q17
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// binary search in a sorted int64 array, return index if found else -1
__device__ __forceinline__ int64_t binary_search(
    const int64_t* __restrict__ arr,
    int64_t n,
    int64_t key)
{
    int64_t lo = 0;
    int64_t hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t val = arr[mid];
        if (val < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < n && arr[lo] == key) return lo;
    return -1;
}

// ------------------------------------------------------------------
// Pass 1: per‑part sum of quantity and count of rows
// ------------------------------------------------------------------
__global__ void q17_pass1(
    const int64_t* __restrict__ l_partkey,
    const double*  __restrict__ l_quantity,
    const int64_t* __restrict__ sorted_part_keys,
    int64_t n_parts,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n_rows)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    int64_t pk = l_partkey[idx];
    int64_t part_idx = binary_search(sorted_part_keys, n_parts, pk);
    if (part_idx >= 0) {
        double q = l_quantity[idx];
        atomicAdd(&sum_qty[part_idx], q);
        atomicAdd(&cnt_qty[part_idx], 1.0);
    }
}

// ------------------------------------------------------------------
// Pass 2: sum extendedprice where quantity < 0.2 * avg_qty per part
// ------------------------------------------------------------------
__global__ void q17_pass2(
    const int64_t* __restrict__ l_partkey,
    const double*  __restrict__ l_quantity,
    const double*  __restrict__ l_extendedprice,
    const int64_t* __restrict__ sorted_part_keys,
    const double*  __restrict__ thresholds,   // 0.2 * avg_qty per part
    int64_t n_parts,
    double* __restrict__ out,
    int64_t n_rows)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    if (idx < n_rows) {
        int64_t pk = l_partkey[idx];
        int64_t part_idx = binary_search(sorted_part_keys, n_parts, pk);
        if (part_idx >= 0) {
            double q = l_quantity[idx];
            double thresh = thresholds[part_idx];
            if (q < thresh) {
                my_rev = l_extendedprice[idx];
            }
        }
    }
    __shared__ double block_sum[BLOCK_SIZE];
    block_sum[threadIdx.x] = my_rev;
    __syncthreads();

    // reduction within block
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}

// ------------------------------------------------------------------
// Launcher: orchestrates the two passes and returns the final result
// ------------------------------------------------------------------
double q17_launcher(
    torch::Tensor l_partkey,          // int64
    torch::Tensor l_quantity,         // float64
    torch::Tensor l_extendedprice,    // float64
    torch::Tensor sorted_part_keys)   // int64, already sorted
{
    TORCH_CHECK(l_partkey.is_cuda(), "l_partkey must be on CUDA");
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be on CUDA");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be on CUDA");
    TORCH_CHECK(sorted_part_keys.is_cuda(), "sorted_part_keys must be on CUDA");

    const int64_t n_rows  = l_partkey.size(0);
    const int64_t n_parts = sorted_part_keys.size(0);

    // allocate per‑part accumulators
    auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor sum_qty = torch::zeros({n_parts}, opts);
    torch::Tensor cnt_qty = torch::zeros({n_parts}, opts);

    // --------------------------------------------------------------
    // Pass 1
    // --------------------------------------------------------------
    const int blocks1 = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q17_pass1<<<blocks1, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        sorted_part_keys.data_ptr<int64_t>(),
        n_parts,
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        n_rows);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("CUDA error (pass1): %s\n", cudaGetErrorString(err1));
    }
    cudaDeviceSynchronize();

    // compute avg and thresholds on the host (small tensors)
    torch::Tensor avg_qty = torch::where(
        cnt_qty > 0.0,
        sum_qty / cnt_qty,
        torch::zeros_like(sum_qty));
    torch::Tensor thresholds = 0.2 * avg_qty;   // 0.2 * avg per part

    // --------------------------------------------------------------
    // Pass 2
    // --------------------------------------------------------------
    torch::Tensor out = torch::zeros({1}, opts);
    const int blocks2 = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q17_pass2<<<blocks2, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        sorted_part_keys.data_ptr<int64_t>(),
        thresholds.data_ptr<double>(),
        n_parts,
        out.data_ptr<double>(),
        n_rows);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA error (pass2): %s\n", cudaGetErrorString(err2));
    }
    cudaDeviceSynchronize();

    double total = out.item<double>();
    return total / 7.0;   // final scaling as per TPC‑H Q17
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_launcher", &q17_launcher, "TPC-H Q17 fused CUDA launcher");
}
'''

def _load_cuda():
    """
    Compile the CUDA extension at import time.
    The source lives in the string `_CUDA_SOURCE` and is written to a
    temporary *.cu file before invoking `torch.utils.cpp_extension.load`.
    """
    tmpdir = tempfile.mkdtemp(prefix="q17_cuda_")
    cu_path = os.path.join(tmpdir, "q17_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q17_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load (once per interpreter lifetime)
_EXT = _load_cuda()

def _launch_q17(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    sorted_part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Thin wrapper around the compiled CUDA launcher.
    Returns a scalar double‑tensor on the same device as the inputs.
    """
    result = _EXT.q17_launcher(l_partkey, l_quantity, l_extendedprice, sorted_part_keys)
    # `result` is a Python float; wrap back as a torch scalar.
    return torch.tensor(result, dtype=torch.float64, device=l_partkey.device)

def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    GPU‑accelerated core for TPC‑H Q17.

    Steps (all on the GPU, fused into two CUDA kernels):
    1. Sort the filtered part keys once (still a cheap GPU sort).
    2. Pass 1 – for each lineitem row, binary‑search the sorted part list.
       If the part belongs to the filtered set, atomically accumulate
       `quantity` and a row count per part.
    3. Compute `avg_qty = sum_qty / cnt_qty` and the 0.2 × average threshold.
    4. Pass 2 – again binary‑search each row; if the row’s quantity is below
       its part’s threshold, atomically add its `extendedprice` to a global sum.
    5. Return `total / 7.0`.

    This replaces the original multi‑kernel torch implementation with a
    fused, memory‑bound pipeline that reads each column only twice and
    performs at most one global atomic per block, yielding higher bandwidth
    utilisation on H100.
    """
    # 1. sort part keys (GPU sort – cheap compared to full scan)
    sorted_keys, _ = torch.sort(part_keys)
    # 2‑5 performed in CUDA
    return _launch_q17(l_partkey, l_quantity, l_extendedprice, sorted_keys)

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
          AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                            FROM lineitem
                            WHERE l_partkey = p_partkey);
    Description: TPC-H Q17. Computes the small‑quantity order revenue average for parts of a
        given brand and container. Correlated subquery is evaluated as a grouped average over
        all lineitems for each filtered partkey.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table first
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
