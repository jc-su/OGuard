import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void q19_kernel(
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int* __restrict__ p_brand_id,
    const int* __restrict__ p_container_grp,
    const int* __restrict__ p_size,
    int64_t n,
    double* __restrict__ out_ptr
) {
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double thread_val = 0.0;
    
    if (idx < n) {
        int brand = p_brand_id[idx];
        int container = p_container_grp[idx];
        int size = p_size[idx];
        double qty = l_quantity[idx];
        double price = l_extendedprice[idx];
        double disc = l_discount[idx];
        
        bool match = false;
        
        // Condition 1: Brand#12 (12), SM (1), Qty 1-11, Size 1-5
        if (brand == 12 && container == 1) {
            if (qty >= 1.0 && qty <= 11.0 && size >= 1 && size <= 5) {
                match = true;
            }
        }
        // Condition 2: Brand#23 (23), MED (2), Qty 10-20, Size 1-10
        else if (brand == 23 && container == 2) {
            if (qty >= 10.0 && qty <= 20.0 && size >= 1 && size <= 10) {
                match = true;
            }
        }
        // Condition 3: Brand#34 (34), LG (3), Qty 20-30, Size 1-15
        else if (brand == 34 && container == 3) {
            if (qty >= 20.0 && qty <= 30.0 && size >= 1 && size <= 15) {
                match = true;
            }
        }
        
        if (match) {
            thread_val = price * (1.0 - disc);
        }
    }
    
    block_sum[threadIdx.x] = thread_val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_sum[0]);
    }
}

double q19_launcher(
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor p_brand_id,
    torch::Tensor p_container_grp,
    torch::Tensor p_size
) {
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(p_brand_id.scalar_type() == torch::kInt32, "p_brand_id must be int32");
    TORCH_CHECK(p_container_grp.scalar_type() == torch::kInt32, "p_container_grp must be int32");
    TORCH_CHECK(p_size.scalar_type() == torch::kInt32, "p_size must be int32");
    
    // Ensure CUDA and contiguous
    auto l_qty = l_quantity.cuda().contiguous();
    auto l_price = l_extendedprice.cuda().contiguous();
    auto l_disc = l_discount.cuda().contiguous();
    auto p_brand = p_brand_id.cuda().contiguous();
    auto p_cont = p_container_grp.cuda().contiguous();
    auto p_sz = p_size.cuda().contiguous();
    
    int64_t n = l_qty.numel();
    auto out = torch::zeros({1}, l_qty.options());
    
    if (n == 0) {
        return 0.0;
    }

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q19_kernel<<<blocks, BLOCK_SIZE>>>(
        l_qty.data_ptr<double>(),
        l_price.data_ptr<double>(),
        l_disc.data_ptr<double>(),
        p_brand.data_ptr<int>(),
        p_cont.data_ptr<int>(),
        p_sz.data_ptr<int>(),
        n,
        out.data_ptr<double>()
    );
    
    // Check for launch errors
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q19_launcher", &q19_launcher, "TPC-H Q19 Revenue (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q19_cuda_")
    cu_path = os.path.join(tmpdir, "_q19_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    
    # H100 is sm_90
    return load(
        name="q19_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

# Load extension at module import
try:
    _EXT = _load_cuda()
except Exception as e:
    raise e

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
    Launches the fused CUDA kernel for filter + project + aggregate.
    The kernel implements the specific Q19 logic internally (hardcoded constants).
    Arguments brand... and ..._grp are kept for API compatibility but unused by kernel.
    """
    res = _EXT.q19_launcher(
        l_quantity,
        l_extendedprice,
        l_discount,
        p_brand_id,
        p_container_grp,
        p_size
    )
    return torch.tensor(res, dtype=torch.float64, device=l_quantity.device)


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
    Description: TPC-H Q19 - Discounted Revenue. Uses cuDF for join/string filtering, 
                 fused CUDA kernel for numeric filtering and aggregation to minimize HBM traffic.
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
    
    if len(joined) == 0:
        return 0.0

    # Encode brand as integer id
    brand12_id, brand23_id, brand34_id = 12, 23, 34
    p_brand_col = joined["p_brand"]
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

    # Convert to torch tensors
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
