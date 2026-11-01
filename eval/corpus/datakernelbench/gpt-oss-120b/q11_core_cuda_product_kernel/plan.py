import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# -------------------------------------------------
# CUDA device code (kernel + launcher) embedded as a string.
# -------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

// Elementwise multiplication: out[i] = a[i] * b[i]
__global__ void product_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] * b[idx];
    }
}

// Launcher returning a new torch::Tensor with the result.
torch::Tensor product_launcher(torch::Tensor a, torch::Tensor b) {
    // Ensure correct type and device
    TORCH_CHECK(a.scalar_type() == torch::kFloat64,
                "ps_supplycost must be torch.float64");
    TORCH_CHECK(b.scalar_type() == torch::kFloat64,
                "ps_availqty must be torch.float64");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "inputs must be CUDA tensors");

    // Make contiguous
    a = a.contiguous();
    b = b.contiguous();

    // Allocate output tensor
    auto out = torch::empty_like(a);

    int64_t N = a.numel();
    int blocks = static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    product_kernel<<<blocks, BLOCK_SIZE>>>(
        a.data_ptr<double>(),
        b.data_ptr<double>(),
        out.data_ptr<double>(),
        N);
    // Propagate any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("product_launcher", &product_launcher,
          "Element‑wise product of two double tensors (CUDA)");
}
'''

def _load_cuda_extension() -> torch.nn.Module:
    """Compile the embedded CUDA source and load it as a PyTorch extension."""
    tmpdir = tempfile.mkdtemp(prefix="q11_cuda_")
    cu_path = os.path.join(tmpdir, "q11_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    # H100 (sm_90) is the target architecture.
    return load(
        name="q11_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load/compile once at import time.
_EXT = _load_cuda_extension()

def _launch_product(
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
) -> torch.Tensor:
    """
    Fuse the element‑wise multiplication of ``ps_supplycost`` and ``ps_availqty``
    into a single CUDA kernel.  The original eager version performed the same
    operation using ``torch.mul`` which also launches a kernel, but this custom
    kernel removes the Python‑level overhead and guarantees coalesced loads.
    """
    return _EXT.product_launcher(ps_supplycost, ps_availqty)


def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core compute for TPC‑H Q11.

    * Fuse the per‑row ``ps_supplycost * ps_availqty`` multiplication via a
      custom CUDA kernel (`_launch_product`).  This reduces launch overhead and
      guarantees a single pass over the two columns.
    * Group by ``ps_partkey`` using ``torch.unique`` / ``scatter_add_`` – both
      are already GPU‑accelerated.
    * Compute the global total, derive the HAVING threshold, filter, and sort
      in descending order of the aggregated value.
    """
    # 1️⃣ Fused multiplication
    value = _launch_product(ps_supplycost, ps_availqty)

    # 2️⃣ Group‑by aggregation (key → sum(value))
    unique_keys, inverse = torch.unique(ps_partkey, return_inverse=True)
    grouped_values = torch.zeros(
        unique_keys.shape[0],
        dtype=value.dtype,
        device=value.device,
    )
    grouped_values.scatter_add_(0, inverse, value)

    # 3️⃣ Global threshold (HAVING)
    total = torch.sum(value)
    threshold = total * threshold_fraction

    # 4️⃣ HAVING filter
    mask = grouped_values > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = grouped_values[mask]

    # 5️⃣ ORDER BY value DESC
    sort_idx = torch.argsort(filtered_values, descending=True)
    return filtered_keys[sort_idx], filtered_values[sort_idx]


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
          AND n_name = 'GERMANY'
        GROUP BY ps_partkey
        HAVING sum(ps_supplycost * ps_availqty) > (
            SELECT sum(ps_supplycost * ps_availqty) * 0.0001
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey
              AND s_nationkey = n_nationkey
              AND n_name = 'GERMANY')
        ORDER BY value DESC;
    Description: TPC‑H Q11 – identifies parts that contribute
                 significantly to the total supply‑cost value for
                 suppliers in a given nation.
    Params:
        nation_name (str): nation name filter literal.
        threshold_fraction (float): fractional threshold used in HAVING clause.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    # Filter nation
    filtered_nation = df_nation[df_nation["n_name"] == nation_name]

    # Join supplier ↔ nation
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )

    # Join partsupp ↔ (supplier ↔ nation)
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")

    # Move columns to GPU tensors (double precision for monetary columns)
    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(
        joined["ps_supplycost"].to_cupy(), dtype=torch.float64
    )
    ps_availqty_t = torch.as_tensor(
        joined["ps_availqty"].to_cupy(), dtype=torch.float64
    )

    # Execute the fused compute core
    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    # Return a cuDF DataFrame matching the reference plan's schema
    return cudf.DataFrame(
        {
            "ps_partkey": keys_t,
            "value": values_t,
        }
    )
