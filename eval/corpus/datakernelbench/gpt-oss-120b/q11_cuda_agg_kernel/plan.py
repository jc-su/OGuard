import os
import tempfile

import torch
import cudf

# ----------------------------------------------------------------------
# CUDA kernel source (compiled at import time)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ------------------------------------------------------------------
// Aggregation kernel: computes value = supplycost * availqty per row,
// atomically adds to per-key sum, and atomically adds to a global total.
// ------------------------------------------------------------------
__global__ void agg_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ supplycost,
    const double* __restrict__ availqty,
    double* __restrict__ groups,
    double* __restrict__ total,
    int64_t n)
{
    // shared memory reduction within a block
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_val = 0.0;

    if (idx < n) {
        my_val = supplycost[idx] * availqty[idx];
        // atomic addition into the per‑key bucket
        atomicAdd(&groups[keys[idx]], my_val);
    }
    block_sum[threadIdx.x] = my_val;
    __syncthreads();

    // intra‑block reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // one atomicAdd per block to the global total
    if (threadIdx.x == 0) {
        atomicAdd(total, block_sum[0]);
    }
}

// ------------------------------------------------------------------
// Launcher: prepares launch configuration, runs kernel, returns total.
// ------------------------------------------------------------------
double agg_launcher(
    torch::Tensor keys,
    torch::Tensor supplycost,
    torch::Tensor availqty,
    torch::Tensor groups)   // groups allocated on device, float64
{
    TORCH_CHECK(keys.is_cuda() && supplycost.is_cuda() && availqty.is_cuda() && groups.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(keys.scalar_type() == torch::kInt64, "keys must be int64");
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64 &&
                availqty.scalar_type() == torch::kFloat64 &&
                groups.scalar_type() == torch::kFloat64,
                "supplycost, availqty, groups must be float64");

    // ensure contiguous memory
    keys = keys.contiguous();
    supplycost = supplycost.contiguous();
    availqty = availqty.contiguous();
    groups = groups.contiguous();

    int64_t n = keys.numel();
    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto total = torch::zeros({1}, supplycost.options());  // float64 on device

    agg_kernel<<<blocks, BLOCK_SIZE>>>(
        keys.data_ptr<int64_t>(),
        supplycost.data_ptr<double>(),
        availqty.data_ptr<double>(),
        groups.data_ptr<double>(),
        total.data_ptr<double>(),
        n);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after kernel: ") + cudaGetErrorString(err));
    }

    return total.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("agg_launcher", &agg_launcher,
          "Fuse multiplication, group‑by sum, and total reduction (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Helper to compile the CUDA extension once at import time
# ----------------------------------------------------------------------
def _load_cuda():
    """Compile the in‑memory CUDA source and return the loaded extension module."""
    tmpdir = tempfile.mkdtemp(prefix="q11_cuda_")
    cu_path = os.path.join(tmpdir, "q11_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)

    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    from torch.utils.cpp_extension import load
    return load(
        name="q11_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load the compiled CUDA extension (cached for the process lifetime)
_EXT = _load_cuda()

# ----------------------------------------------------------------------
# Core query logic (GPU‑accelerated aggregation)
# ----------------------------------------------------------------------
def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU‑fused aggregation for TPC‑H Q11.

    1. Compute ``value = ps_supplycost * ps_availqty`` per row.
    2. Atomically accumulate ``value`` into a per‑partkey buffer.
    3. Simultaneously compute the grand total (used for the HAVING threshold).

    After the kernel we filter, extract, and sort the results.

    Returns
    -------
    (keys, values) : torch.Tensor
        Keys are int64 part keys that passed the HAVING filter, sorted.
        Values are the corresponding summed ``value`` (float64), sorted DESC.
    """
    device = ps_partkey.device

    # Allocate per‑key accumulation buffer; size = max key + 1 (dense keys)
    max_key = int(torch.max(ps_partkey).item())
    groups = torch.zeros(max_key + 1, dtype=torch.float64, device=device)

    # Run the fused kernel; it returns the grand total as a Python float
    total = _EXT.agg_launcher(ps_partkey, ps_supplycost, ps_availqty, groups)

    # HAVING threshold
    threshold = total * threshold_fraction

    # Apply HAVING filter
    mask = groups > threshold
    filtered_keys = torch.nonzero(mask, as_tuple=False).squeeze(1)
    filtered_vals = groups[mask]

    # Order by value descending
    sort_idx = torch.argsort(filtered_vals, descending=True)
    return filtered_keys[sort_idx], filtered_vals[sort_idx]

# ----------------------------------------------------------------------
# Public API (unchanged signature)
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "GERMANY",
    threshold_fraction: float = 0.0001,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey
          AND s_nationkey = n_nationkey
          AND n_name = '<nation_name>'
        GROUP BY ps_partkey
        HAVING sum(ps_supplycost * ps_availqty) > (
            SELECT sum(ps_supplycost * ps_availqty) * <threshold_fraction>
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey
              AND s_nationkey = n_nationkey
              AND n_name = '<nation_name>'
        )
        ORDER BY value DESC;

    Description:
        Joins are performed with cuDF (nation ↔ supplier ↔ partsupp).  The
        heavy aggregation (multiplication, group‑by sum, and total reduction)
        is executed on the GPU via a single fused CUDA kernel, reducing memory
        traffic and kernel launch overhead.  This delivers a faster execution
        on H100 while preserving exact float64 results.

    Params
    ------
    tables : dict[str, cudf.DataFrame]
        Mapping of table name to cuDF DataFrame.
    nation_name : str, default "GERMANY"
        Nation filter literal.
    threshold_fraction : float, default 0.0001
        Fraction used in the HAVING clause.

    Returns
    -------
    cudf.DataFrame
        Columns ``ps_partkey`` (int64) and ``value`` (float64) sorted descending.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    # 1️⃣ Filter nation by name
    filtered_nation = df_nation[df_nation["n_name"] == nation_name]

    # 2️⃣ Join supplier ↔ nation
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )

    # 3️⃣ Join partsupp ↔ (supplier ↔ nation)
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")

    # 4️⃣ Materialize needed columns as GPU tensors (float64 for monetary values)
    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)

    # 5️⃣ GPU‑accelerated aggregation
    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    # 6️⃣ Return cuDF result
    return cudf.DataFrame({"ps_partkey": keys_t, "value": values_t})
