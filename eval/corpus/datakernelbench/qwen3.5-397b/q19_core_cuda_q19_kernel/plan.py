import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void q19_kernel(
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ brand_id_ptr,
    const int32_t* __restrict__ container_grp_ptr,
    const int32_t* __restrict__ size_ptr,
    int32_t brand12_id,
    int32_t brand23_id,
    int32_t brand34_id,
    int32_t sm_grp,
    int32_t md_grp,
    int32_t lg_grp,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (idx < n_elements) {
        int32_t brand_id = brand_id_ptr[idx];
        int32_t cont_grp = container_grp_ptr[idx];
        int32_t p_size = size_ptr[idx];
        double l_qty = quantity_ptr[idx];
        double l_price = price_ptr[idx];
        double l_disc = discount_ptr[idx];
        
        bool cond1 = (brand_id == brand12_id) 
                  && (cont_grp == sm_grp)
                  && (l_qty >= 1.0) && (l_qty <= 11.0)
                  && (p_size >= 1) && (p_size <= 5);
        
        bool cond2 = (brand_id == brand23_id)
                  && (cont_grp == md_grp)
                  && (l_qty >= 10.0) && (l_qty <= 20.0)
                  && (p_size >= 1) && (p_size <= 10);
        
        bool cond3 = (brand_id == brand34_id)
                  && (cont_grp == lg_grp)
                  && (l_qty >= 20.0) && (l_qty <= 30.0)
                  && (p_size >= 1) && (p_size <= 15);
        
        if (cond1 || cond2 || cond3) {
            my_rev = l_price * (1.0 - l_disc);
        }
    }
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }
    
    // One atomicAdd per block
    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}

double q19_launcher(
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor p_brand_id,
    torch::Tensor p_container_grp,
    torch::Tensor p_size,
    int32_t brand12_id,
    int32_t brand23_id,
    int32_t brand34_id,
    int32_t sm_grp,
    int32_t md_grp,
    int32_t lg_grp
) {
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(p_brand_id.scalar_type() == torch::kInt32, "p_brand_id must be int32");
    TORCH_CHECK(p_container_grp.scalar_type() == torch::kInt32, "p_container_grp must be int32");
    TORCH_CHECK(p_size.scalar_type() == torch::kInt32, "p_size must be int32");
    
    // Ensure CUDA and contiguous
    if (!l_quantity.is_cuda()) l_quantity = l_quantity.cuda();
    if (!l_extendedprice.is_cuda()) l_extendedprice = l_extendedprice.cuda();
    if (!l_discount.is_cuda()) l_discount = l_discount.cuda();
    if (!p_brand_id.is_cuda()) p_brand_id = p_brand_id.cuda();
    if (!p_container_grp.is_cuda()) p_container_grp = p_container_grp.cuda();
    if (!p_size.is_cuda()) p_size = p_size.cuda();
    
    l_quantity = l_quantity.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    p_brand_id = p_brand_id.contiguous();
    p_container_grp = p_container_grp.contiguous();
    p_size = p_size.contiguous();
    
    int64_t n = l_quantity.numel();
    auto out = torch::zeros({1}, l_quantity.options());
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q19_kernel<<<blocks, BLOCK_SIZE>>>(
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        p_brand_id.data_ptr<int32_t>(),
        p_container_grp.data_ptr<int32_t>(),
        p_size.data_ptr<int32_t>(),
        brand12_id, brand23_id, brand34_id,
        sm_grp, md_grp, lg_grp,
        out.data_ptr<double>(),
        n
    );
    
    cudaDeviceSynchronize();
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q19_launcher", &q19_launcher, "TPC-H Q19 revenue (CUDA fused kernel)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q19_cuda_")
    cu_path = os.path.join(tmpdir, "_q19_kernels.cu")
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
    Launch CUDA kernel for Q19 revenue calculation.
    
    GPU: Single fused kernel for filter → project → aggregate.
    Why faster: One HBM pass instead of multiple cuDF/PyTorch passes,
    block-level reduction minimizes global atomics.
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
    TPC-H Q19 core computation: discounted revenue with multi-disjunction predicates.
    
    GPU: CUDA fused kernel (filter + project + aggregate in one pass).
    Why faster: Memory-bound scan optimized via fusion - single HBM read,
    block shared-memory reduction, one atomicAdd per block instead of per-thread.
    
    Args:
        l_quantity: Line item quantities (float64)
        l_extendedprice: Line item extended prices (float64)
        l_discount: Line item discounts (float64)
        p_brand_id: Encoded brand IDs (int32: 12, 23, 34)
        p_container_grp: Encoded container groups (int32: 1=SM, 2=MED, 3=LG)
        p_size: Part sizes (int32)
        brand12_id, brand23_id, brand34_id: Brand ID constants
        sm_grp, md_grp, lg_grp: Container group constants
    
    Returns:
        Scalar tensor with total revenue (float64)
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
    Description: TPC-H Q19 - Discounted Revenue with multi-disjunction predicates on part/lineitem join.
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
    p_brand_id = cudf.Series(
        [0] * len(joined), dtype="int32"
    )
    # Use where-based encoding via masks
    brand_encoded = cudf.Series([0] * len(joined), dtype="int32")
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

    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(brand_encoded.to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(container_grp.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

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
