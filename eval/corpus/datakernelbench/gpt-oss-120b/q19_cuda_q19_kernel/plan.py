import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: one fused kernel that evaluates the three disjunctive
# predicates, computes the revenue term, and reduces the result.
# It is compiled at import time via torch.utils.cpp_extension.load().
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// One‑dimensional kernel: each thread processes one row.
__global__ void q19_kernel(
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int*    __restrict__ p_brand_id,
    const int*    __restrict__ p_container_grp,
    const int*    __restrict__ p_size,
    const int     brand12_id,
    const int     brand23_id,
    const int     brand34_id,
    const int     sm_grp,
    const int     md_grp,
    const int     lg_grp,
    double* __restrict__ out,
    const int64_t N)
{
    __shared__ double sdata[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;

    if (idx < N) {
        // Load columns (coalesced)
        double qty   = l_quantity[idx];
        double price = l_extendedprice[idx];
        double disc  = l_discount[idx];
        int    brand = p_brand_id[idx];
        int    cont  = p_container_grp[idx];
        int    size  = p_size[idx];

        // ----- three disjunctive predicates -------------------------------------------------
        bool cond1 = (brand == brand12_id) && (cont == sm_grp) &&
                     (qty >= 1.0) && (qty <= 11.0) &&
                     (size >= 1) && (size <= 5);
        bool cond2 = (brand == brand23_id) && (cont == md_grp) &&
                     (qty >= 10.0) && (qty <= 20.0) &&
                     (size >= 1) && (size <= 10);
        bool cond3 = (brand == brand34_id) && (cont == lg_grp) &&
                     (qty >= 20.0) && (qty <= 30.0) &&
                     (size >= 1) && (size <= 15);
        // ------------------------------------------------------------------------------------
        if (cond1 || cond2 || cond3) {
            my_rev = price * (1.0 - disc);
        }
    }

    // Block‑level reduction in shared memory
    sdata[threadIdx.x] = my_rev;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // One atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// Launcher: wraps the kernel and returns the reduced sum as a double.
double q19_launcher(
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor p_brand_id,
    torch::Tensor p_container_grp,
    torch::Tensor p_size,
    const int brand12_id,
    const int brand23_id,
    const int brand34_id,
    const int sm_grp,
    const int md_grp,
    const int lg_grp)
{
    // All tensors must be on CUDA and contiguous.
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be a CUDA tensor");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be a CUDA tensor");
    TORCH_CHECK(l_discount.is_cuda(), "l_discount must be a CUDA tensor");
    TORCH_CHECK(p_brand_id.is_cuda(), "p_brand_id must be a CUDA tensor");
    TORCH_CHECK(p_container_grp.is_cuda(), "p_container_grp must be a CUDA tensor");
    TORCH_CHECK(p_size.is_cuda(), "p_size must be a CUDA tensor");

    // Ensure correct dtypes.
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(p_brand_id.scalar_type() == torch::kInt32, "p_brand_id must be int32");
    TORCH_CHECK(p_container_grp.scalar_type() == torch::kInt32, "p_container_grp must be int32");
    TORCH_CHECK(p_size.scalar_type() == torch::kInt32, "p_size must be int32");

    // Make contiguous views.
    l_quantity = l_quantity.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    p_brand_id = p_brand_id.contiguous();
    p_container_grp = p_container_grp.contiguous();
    p_size = p_size.contiguous();

    int64_t N = l_quantity.numel();
    // Output buffer (single double) in device memory.
    auto out = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

    int blocks = static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid(blocks);
    dim3 block(BLOCK_SIZE);

    // Launch kernel
    q19_kernel<<<grid, block>>>(
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        p_brand_id.data_ptr<int>(),
        p_container_grp.data_ptr<int>(),
        p_size.data_ptr<int>(),
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
        out.data_ptr<double>(),
        N
    );

    // Wait for kernel completion and propagate errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // Copy the result back to host and return.
    double host_val = out.item<double>();
    return host_val;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q19_launcher", &q19_launcher, "TPC‑H Q19 fused filter/aggregate (CUDA)");
}
'''

def _load_cuda_extension() -> torch.nn.Module:
    """Compile the CUDA extension from the in‑memory source string."""
    tmpdir = tempfile.mkdtemp(prefix="q19_cuda_")
    cu_path = os.path.join(tmpdir, "q19_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    # Build with -arch=sm_90 for H100.
    return load(
        name="q19_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Compile once at import time.
_CUDA_EXT = _load_cuda_extension()


def _query_core(
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    p_brand_id: torch.Tensor,
    p_container_grp: torch.Tensor,
    p_size: torch.Tensor,
    brand12_id: int,
    brand23_id: int,
    brand34_id: int,
    sm_grp: int,
    md_grp: int,
    lg_grp: int,
) -> torch.Tensor:
    """
    GPU‑accelerated core for TPC‑H Q19.

    All input tensors must be on the CUDA device and have the dtypes:
        l_quantity, l_extendedprice, l_discount : float64
        p_brand_id, p_container_grp, p_size   : int32

    The function returns a zero‑dimensional ``torch.Tensor`` (float64) containing
    the summed revenue computed by the fused CUDA kernel.
    """
    # The launcher returns a Python ``float``; wrap it back into a tensor for
    # consistency with the reference API (which expects a torch.Tensor).
    result = _CUDA_EXT.q19_launcher(
        l_quantity,
        l_extendedprice,
        l_discount,
        p_brand_id,
        p_container_grp,
        p_size,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
    )
    # ``result`` is a Python float; convert to a scalar tensor.
    return torch.tensor(result, dtype=torch.float64, device=l_quantity.device)


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand1: str = "Brand#12",
    brand2: str = "Brand#23",
    brand3: str = "Brand#34",
    shipinstruct: str = "DELIVER IN PERSON",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue
        FROM lineitem, part
        WHERE (p_partkey = l_partkey AND p_brand = 'Brand#12'
               AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
               AND l_quantity BETWEEN 1 AND 11
               AND p_size BETWEEN 1 AND 5
               AND l_shipmode IN ('AIR','AIR REG')
               AND l_shipinstruct = 'DELIVER IN PERSON')
           OR (p_brand = 'Brand#23' AND p_container IN ('MED BAG','MED BOX','MED PKG','MED PACK')
               AND l_quantity BETWEEN 10 AND 20
               AND p_size BETWEEN 1 AND 10
               AND l_shipmode IN ('AIR','AIR REG')
               AND l_shipinstruct = 'DELIVER IN PERSON')
           OR (p_brand = 'Brand#34' AND p_container IN ('LG CASE','LG BOX','LG PACK','LG PKG')
               AND l_quantity BETWEEN 20 AND 30
               AND p_size BETWEEN 1 AND 15
               AND l_shipmode IN ('AIR','AIR REG')
               AND l_shipinstruct = 'DELIVER IN PERSON');

    Description:
        The join and string handling are performed with cuDF (which is already
        highly optimized for such operations).  After the join, the
        three‑disjunction predicate together with the revenue projection are
        fused into a single CUDA kernel above.  This eliminates multiple
        materializations, reduces memory traffic, and performs a single
        block‑wise reduction, yielding a faster execution on H100.

    Params:
        brand1, brand2, brand3 (str): brand literals.
        shipinstruct (str): ship instruction literal.
    Returns:
        revenue (float)
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # ------------------------------------------------------------------
    # 1) Filter lineitem on shipmode & ship instruction (common to all paths)
    # ------------------------------------------------------------------
    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]

    # ------------------------------------------------------------------
    # 2) Filter part on the three brands (common narrowing)
    # ------------------------------------------------------------------
    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part[part_mask]

    # ------------------------------------------------------------------
    # 3) Join on partkey
    # ------------------------------------------------------------------
    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    # ------------------------------------------------------------------
    # 4) Encode brand as small integers (12, 23, 34)
    # ------------------------------------------------------------------
    p_brand_col = joined["p_brand"]
    brand12_id, brand23_id, brand34_id = 12, 23, 34
    brand_encoded = (
        (p_brand_col == brand1).astype("int32") * brand12_id
        + (p_brand_col == brand2).astype("int32") * brand23_id
        + (p_brand_col == brand3).astype("int32") * brand34_id
    )

    # ------------------------------------------------------------------
    # 5) Encode container group: 1 = SM, 2 = MED, 3 = LG, 0 = other
    # ------------------------------------------------------------------
    sm_grp, md_grp, lg_grp = 1, 2, 3
    cont = joined["p_container"]
    sm_mask = (
        (cont == "SM CASE") | (cont == "SM BOX") | (cont == "SM PACK") | (cont == "SM PKG")
    ).astype("int32") * sm_grp
    md_mask = (
        (cont == "MED BAG") | (cont == "MED BOX") | (cont == "MED PKG") | (cont == "MED PACK")
    ).astype("int32") * md_grp
    lg_mask = (
        (cont == "LG CASE") | (cont == "LG BOX") | (cont == "LG PACK") | (cont == "LG PKG")
    ).astype("int32") * lg_grp
    container_grp = sm_mask + md_mask + lg_mask

    # ------------------------------------------------------------------
    # 6) Move numeric columns to CUDA tensors (float64 for FP columns)
    # ------------------------------------------------------------------
    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(brand_encoded.to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(container_grp.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    # ------------------------------------------------------------------
    # 7) Execute fused GPU kernel
    # ------------------------------------------------------------------
    result_tensor = _query_core(
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        p_brand_id_t,
        p_container_grp_t,
        p_size_t,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
    )
    # Convert to Python float for the public API.
    return float(result_tensor.item())
