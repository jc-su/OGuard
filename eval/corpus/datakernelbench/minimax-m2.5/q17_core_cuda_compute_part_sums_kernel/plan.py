import os
import tempfile
import torch
from torch.utils.cpp_extension import load
import cudf

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

// ----------------------------------------------------------------------
// Kernel 1: compute per‑part sum of quantities and counts using atomics
// ----------------------------------------------------------------------
__global__ void compute_part_sums_kernel(
    const int64_t* __restrict__ part_idx,
    const double* __restrict__ quantity,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int64_t p = part_idx[tid];
        double q = quantity[tid];
        atomicAdd(&sum_qty[p], q);
        atomicAdd(&cnt_qty[p], 1.0);
    }
}

// ----------------------------------------------------------------------
// Kernel 2: block‑reduced filter + aggregation of extendedprice
// ----------------------------------------------------------------------
__global__ void compute_filtered_sum_kernel(
    const int64_t* __restrict__ part_idx,
    const double* __restrict__ quantity,
    const double* __restrict__ price,
    const double* __restrict__ threshold,
    double* __restrict__ total,
    int64_t n
) {
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double my_val = 0.0;
    if (tid < n) {
        int64_t p = part_idx[tid];
        double q = quantity[tid];
        if (q < threshold[p]) {
            my_val = price[tid];
        }
    }
    block_sum[threadIdx.x] = my_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(total, block_sum[0]);
    }
}

// ----------------------------------------------------------------------
// Launchers (exposed via PyBind)
// ----------------------------------------------------------------------
void compute_part_sums(
    torch::Tensor part_idx,
    torch::Tensor quantity,
    torch::Tensor sum_qty,
    torch::Tensor cnt_qty
) {
    TORCH_CHECK(part_idx.is_cuda(), "part_idx must be a CUDA tensor");
    TORCH_CHECK(quantity.is_cuda(), "quantity must be a CUDA tensor");
    TORCH_CHECK(sum_qty.is_cuda(), "sum_qty must be a CUDA tensor");
    TORCH_CHECK(cnt_qty.is_cuda(), "cnt_qty must be a CUDA tensor");
    TORCH_CHECK(part_idx.scalar_type() == torch::kInt64, "part_idx must be int64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(sum_qty.scalar_type() == torch::kFloat64, "sum_qty must be float64");
    TORCH_CHECK(cnt_qty.scalar_type() == torch::kFloat64, "cnt_qty must be float64");

    part_idx = part_idx.contiguous();
    quantity = quantity.contiguous();

    int64_t n = part_idx.numel();
    int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_part_sums_kernel<<<blocks, BLOCK_SIZE>>>(
        part_idx.data_ptr<int64_t>(),
        quantity.data_ptr<double>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        n
    );
    cudaDeviceSynchronize();
}

void compute_filtered_sum(
    torch::Tensor part_idx,
    torch::Tensor quantity,
    torch::Tensor price,
    torch::Tensor threshold,
    torch::Tensor total
) {
    TORCH_CHECK(part_idx.is_cuda(), "part_idx must be a CUDA tensor");
    TORCH_CHECK(quantity.is_cuda(), "quantity must be a CUDA tensor");
    TORCH_CHECK(price.is_cuda(), "price must be a CUDA tensor");
    TORCH_CHECK(threshold.is_cuda(), "threshold must be a CUDA tensor");
    TORCH_CHECK(total.is_cuda(), "total must be a CUDA tensor");
    TORCH_CHECK(part_idx.scalar_type() == torch::kInt64, "part_idx must be int64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(threshold.scalar_type() == torch::kFloat64, "threshold must be float64");
    TORCH_CHECK(total.scalar_type() == torch::kFloat64, "total must be float64");

    part_idx = part_idx.contiguous();
    quantity = quantity.contiguous();
    price = price.contiguous();
    threshold = threshold.contiguous();

    int64_t n = part_idx.numel();
    int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_filtered_sum_kernel<<<blocks, BLOCK_SIZE>>>(
        part_idx.data_ptr<int64_t>(),
        quantity.data_ptr<double>(),
        price.data_ptr<double>(),
        threshold.data_ptr<double>(),
        total.data_ptr<double>(),
        n
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_part_sums", &compute_part_sums,
          "Compute per‑part sum of quantities and counts (CUDA)");
    m.def("compute_filtered_sum", &compute_filtered_sum,
          "Compute filtered sum of extendedprice (CUDA)");
}
'''


def _load_cuda():
    """Write the CUDA source to a temporary file and JIT‑compile it."""
    tmpdir = tempfile.mkdtemp(prefix="q17_cuda_")
    cu_path = os.path.join(tmpdir, "_q17_kernels.cu")
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
    )


_EXT = _load_cuda()


def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC‑H Q17: sum(l_extendedprice)/7.0 where
    l_quantity < 0.2 * avg(l_quantity) for each filtered part.

    This implementation uses two fused CUDA kernels:
    1. A kernel that atomically accumulates per‑part quantity sums and counts.
    2. A block‑reduced kernel that applies the per‑part threshold and aggregates
       the final revenue.  The second kernel reduces atomic contention by
       performing a block‑level reduction before a single atomicAdd per block.

    The host side performs the sorting / searchsorted mapping and the cheap
    per‑part average / threshold computation.
    """
    # ------------------------------------------------------------------
    # Build a dense mapping from filtered part keys to [0 .. n_parts-1]
    # ------------------------------------------------------------------
    sorted_keys, _ = torch.sort(part_keys)
    n_parts = sorted_keys.numel()

    # Locate each lineitem's partkey inside the sorted filtered keys
    idx = torch.searchsorted(sorted_keys, l_partkey)
    idx_clamped = torch.clamp(idx, max=n_parts - 1)
    in_part = sorted_keys[idx_clamped] == l_partkey

    # Keep only lineitems whose partkey appears in the filtered part set
    sel_idx = idx_clamped[in_part]           # int64 index into sorted_keys
    sel_qty = l_quantity[in_part]             # double
    sel_price = l_extendedprice[in_part]      # double

    # ------------------------------------------------------------------
    # Kernel 1: accumulate per‑part quantity sums and counts
    # ------------------------------------------------------------------
    sum_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_partkey.device)
    cnt_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_partkey.device)

    _EXT.compute_part_sums(sel_idx, sel_qty, sum_qty, cnt_qty)

    # ------------------------------------------------------------------
    # Compute per‑part average and threshold (0.2 * average)
    # ------------------------------------------------------------------
    avg_qty = torch.where(cnt_qty > 0, sum_qty / cnt_qty, torch.zeros_like(sum_qty))
    threshold = 0.2 * avg_qty  # double

    # ------------------------------------------------------------------
    # Kernel 2: apply the per‑part threshold and sum extendedprice
    # ------------------------------------------------------------------
    total = torch.zeros(1, dtype=torch.float64, device=l_partkey.device)
    _EXT.compute_filtered_sum(sel_idx, sel_qty, sel_price, threshold, total)

    return total / 7.0


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
    Description: TPC-H Q17. Computes the small-quantity order revenue average for
        parts of a given brand and container. The correlated subquery is evaluated
        as a grouped average over all lineitems for each filtered partkey.
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
