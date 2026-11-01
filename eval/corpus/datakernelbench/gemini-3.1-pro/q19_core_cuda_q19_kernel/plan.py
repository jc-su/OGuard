import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q19_kernel(
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ extendedprice_ptr,
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
    int64_t numel
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (idx < numel) {
        int32_t brand = brand_id_ptr[idx];
        int32_t container = container_grp_ptr[idx];
        
        bool c1 = (brand == brand12_id) && (container == sm_grp);
        bool c2 = (brand == brand23_id) && (container == md_grp);
        bool c3 = (brand == brand34_id) && (container == lg_grp);
        
        if (c1 || c2 || c3) {
            double qty = quantity_ptr[idx];
            int32_t sz = size_ptr[idx];
            bool keep = false;
            
            if (c1 && qty >= 1.0 && qty <= 11.0 && sz >= 1 && sz <= 5) keep = true;
            else if (c2 && qty >= 10.0 && qty <= 20.0 && sz >= 1 && sz <= 10) keep = true;
            else if (c3 && qty >= 20.0 && qty <= 30.0 && sz >= 1 && sz <= 15) keep = true;
            
            if (keep) {
                double ep = extendedprice_ptr[idx];
                double disc = discount_ptr[idx];
                my_rev = ep * (1.0 - disc);
            }
        }
    }
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}

double q19_launcher(
    torch::Tensor quantity,
    torch::Tensor extendedprice,
    torch::Tensor discount,
    torch::Tensor brand_id,
    torch::Tensor container_grp,
    torch::Tensor size,
    int32_t brand12_id,
    int32_t brand23_id,
    int32_t brand34_id,
    int32_t sm_grp,
    int32_t md_grp,
    int32_t lg_grp
) {
    TORCH_CHECK(quantity.is_cuda(), "quantity must be a CUDA tensor");
    TORCH_CHECK(extendedprice.is_cuda(), "extendedprice must be a CUDA tensor");
    TORCH_CHECK(discount.is_cuda(), "discount must be a CUDA tensor");
    TORCH_CHECK(brand_id.is_cuda(), "brand_id must be a CUDA tensor");
    TORCH_CHECK(container_grp.is_cuda(), "container_grp must be a CUDA tensor");
    TORCH_CHECK(size.is_cuda(), "size must be a CUDA tensor");
    
    quantity = quantity.contiguous();
    extendedprice = extendedprice.contiguous();
    discount = discount.contiguous();
    brand_id = brand_id.contiguous();
    container_grp = container_grp.contiguous();
    size = size.contiguous();

    int64_t numel = quantity.numel();
    auto out = torch::zeros({1}, quantity.options());
    
    if (numel > 0) {
        int threads = BLOCK_SIZE;
        int blocks = (numel + threads - 1) / threads;
        
        q19_kernel<<<blocks, threads>>>(
            quantity.data_ptr<double>(),
            extendedprice.data_ptr<double>(),
            discount.data_ptr<double>(),
            brand_id.data_ptr<int32_t>(),
            container_grp.data_ptr<int32_t>(),
            size.data_ptr<int32_t>(),
            brand12_id,
            brand23_id,
            brand34_id,
            sm_grp,
            md_grp,
            lg_grp,
            out.data_ptr<double>(),
            numel
        );
    }
    
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q19_launcher", &q19_launcher, "TPC-H Q19 Kernel");
}
'''

def _load_cuda() -> object:
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
    Computes TPC-H Q19 revenue over already-joined vectors using a fused CUDA kernel.
    By short-circuiting the evaluation in a single pass, it is faster and requires fewer memory operations, 
    effectively avoiding multiple boolean temporary mask tensors to compute conditions.
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
        lg_grp
    )
    return torch.tensor(rev, dtype=torch.float64, device=l_quantity.device)


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
