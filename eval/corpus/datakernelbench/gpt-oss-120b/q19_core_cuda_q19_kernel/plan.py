import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA device code (kernel + launcher) embedded as a string.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// fused kernel for TPC-H Q19 revenue computation
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
    double*       __restrict__ out,
    const int64_t n) {

    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    double my_rev = 0.0;

    if (idx < n) {
        // Load once per thread (coalesced)
        double qty   = l_quantity[idx];
        double price = l_extendedprice[idx];
        double disc  = l_discount[idx];
        int    brand = p_brand_id[idx];
        int    cont  = p_container_grp[idx];
        int    size  = p_size[idx];

        bool keep = false;

        // condition 1
        if (brand == brand12_id && cont == sm_grp &&
            qty >= 1.0 && qty <= 11.0 &&
            size >= 1 && size <= 5) {
            keep = true;
        }
        // condition 2
        if (!keep && brand == brand23_id && cont == md_grp &&
            qty >= 10.0 && qty <= 20.0 &&
            size >= 1 && size <= 10) {
            keep = true;
        }
        // condition 3
        if (!keep && brand == brand34_id && cont == lg_grp &&
            qty >= 20.0 && qty <= 30.0 &&
            size >= 1 && size <= 15) {
            keep = true;
        }

        if (keep) {
            my_rev = price * (1.0 - disc);
        }
    }

    // block‑wise reduction
    block_sum[threadIdx.x] = my_rev;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // one atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}

// C++ launcher exposing the kernel to Python
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
    const int lg_grp) {

    // sanity checks – all tensors must be on CUDA, contiguous, correct dtype
    TORCH_CHECK(l_quantity.is_cuda() && l_quantity.dtype() == torch::kFloat64,
                "l_quantity must be CUDA double");
    TORCH_CHECK(l_extendedprice.is_cuda() && l_extendedprice.dtype() == torch::kFloat64,
                "l_extendedprice must be CUDA double");
    TORCH_CHECK(l_discount.is_cuda() && l_discount.dtype() == torch::kFloat64,
                "l_discount must be CUDA double");
    TORCH_CHECK(p_brand_id.is_cuda() && p_brand_id.dtype() == torch::kInt32,
                "p_brand_id must be CUDA int32");
    TORCH_CHECK(p_container_grp.is_cuda() && p_container_grp.dtype() == torch::kInt32,
                "p_container_grp must be CUDA int32");
    TORCH_CHECK(p_size.is_cuda() && p_size.dtype() == torch::kInt32,
                "p_size must be CUDA int32");

    // ensure contiguous
    l_quantity       = l_quantity.contiguous();
    l_extendedprice  = l_extendedprice.contiguous();
    l_discount       = l_discount.contiguous();
    p_brand_id       = p_brand_id.contiguous();
    p_container_grp  = p_container_grp.contiguous();
    p_size           = p_size.contiguous();

    int64_t n = l_quantity.numel();

    // output buffer (single double)
    auto out = torch::zeros({1}, l_quantity.options());

    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q19_kernel<<<blocks, BLOCK_SIZE>>>(
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
        n
    );

    // synchronize and propagate possible launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch error: ", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q19_launcher", &q19_launcher,
          "Fused Q19 revenue computation (CUDA)");
}
'''

def _load_cuda():
    """Write the CUDA source to a temporary file and compile it with
    ``torch.utils.cpp_extension.load``.  The resulting module is cached for
    the lifetime of the Python process."""
    tmpdir = tempfile.mkdtemp(prefix="q19_cuda_")
    cu_path = os.path.join(tmpdir, "q19_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q19_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

# compiled extension (loaded once at import)
_EXT = _load_cuda()

def _launch_q19(
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
    Thin Python wrapper around the compiled CUDA launcher.
    Returns a zero‑dim tensor on the device containing the revenue.
    """
    rev = _EXT.q19_launcher(
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
    # ``rev`` is a Python float; wrap as a scalar CUDA tensor for API parity
    return torch.tensor(rev, dtype=torch.float64, device=l_quantity.device)

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
    GPU‑accelerated core of TPC‑H Q19.  All column tensors are 1‑D, already
    materialised on the device.  The function fuses the three disjunctive
    predicates, the revenue projection, and the global sum into a single
    CUDA kernel, eliminating intermediate temporaries and reducing HBM traffic.
    """
    return _launch_q19(
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

def run_query(
    tables: dict[str, cudf.DataFrame],
    brand1: str = "Brand#12",
    brand2: str = "Brand#23",
    brand3: str = "Brand#34",
    shipinstruct: str = "DELIVER IN PERSON",
) -> float:
    """
    SQL: SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE
         (p_partkey=l_partkey AND p_brand='Brand#12' AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
          AND l_quantity BETWEEN 1 AND 11 AND p_size BETWEEN 1 AND 5
          AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct='DELIVER IN PERSON')
         OR (... Brand#23, MED containers, qty 10-20, size 1-10 ...)
         OR (... Brand#34, LG containers, qty 20-30, size 1-15 ...).
    Description: TPC‑H Q19 - Discounted Revenue with multi‑disjunction predicates on part/lineitem join.
    Params:
        brand1, brand2, brand3 (str): the three brand literals.
        shipinstruct (str): ship instruction literal.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Pre-filter lineitem on shipmode/shipinstruct (common to all disjuncts)
    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]

    # Pre-filter part on brand set (common narrowing)
    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part[part_mask]

    # Join on partkey
    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    # Encode brand as integer id
    brand12_id, brand23_id, brand34_id = 12, 23, 34
    p_brand_col = joined["p_brand"]
    # one‑hot encoding via masks (int32)
    b12 = (p_brand_col == brand1).astype("int32") * brand12_id
    b23 = (p_brand_col == brand2).astype("int32") * brand23_id
    b34 = (p_brand_col == brand3).astype("int32") * brand34_id
    brand_encoded = b12 + b23 + b34

    # Encode container into groups: 1=SM, 2=MED, 3=LG, 0=other
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

    # Convert cuDF columns to CUDA tensors (float64 for numeric, int32 for ids)
    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(brand_encoded.to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(container_grp.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    # Execute fused CUDA kernel
    res = _query_core(
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
    return float(res.item())
