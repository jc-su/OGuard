import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q19_kernel(
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ p_brand_id,
    const int32_t* __restrict__ p_container_grp,
    const int32_t* __restrict__ p_size,
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
        int32_t brand = p_brand_id[idx];
        int32_t container = p_container_grp[idx];
        double qty = l_quantity[idx];
        int32_t size = p_size[idx];
        bool cond1 = (brand == brand12_id) && (container == sm_grp) && (qty >= 1.0) && (qty <= 11.0) && (size >= 1) && (size <= 5);
        bool cond2 = (brand == brand23_id) && (container == md_grp) && (qty >= 10.0) && (qty <= 20.0) && (size >= 1) && (size <= 10);
        bool cond3 = (brand == brand34_id) && (container == lg_grp) && (qty >= 20.0) && (qty <= 30.0) && (size >= 1) && (size <= 15);
        if (cond1 || cond2 || cond3) {
            my_rev = l_extendedprice[idx] * (1.0 - l_discount[idx]);
        }
    }
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
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
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64);
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
        brand12_id, brand23_id, brand34_id, sm_grp, md_grp, lg_grp,
        out.data_ptr<double>(), n
    );
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q19_launcher", &q19_launcher, "TPC-H Q19 revenue (CUDA)");
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
    GPU Optimization: Fused CUDA kernel reduces HBM traffic by: 1) one coalesced 1D pass over all filtered-but-not-tiny joined rows, 2) conditional load of only l_extendedprice/l_discount when the (brand,container,qty,size) predicate fires, 3) block-level shared reduce + single atomicAdd per block.
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
    p_brand_id = cudf.Series([0] * len(joined), dtype="int32")
    b12 = (joined["p_brand"] == brand1).astype("int32") * brand12_id
    b23 = (joined["p_brand"] == brand2).astype("int32") * brand23_id
    b34 = (joined["p_brand"] == brand3).astype("int32") * brand34_id
    p_brand_id = b12 + b23 + b34

    # Encode container into groups: 1=SM, 2=MED, 3=LG, 0=other
    sm_grp, md_grp, lg_grp = 1, 2, 3
    container_grp = cudf.Series([0] * len(joined), dtype="int32")
    sm_mask = (
        (joined["p_container"] == "SM CASE") | (joined["p_container"] == "SM BOX") |
        (joined["p_container"] == "SM PACK") | (joined["p_container"] == "SM PKG")
    ).astype("int32") * sm_grp
    md_mask = (
        (joined["p_container"] == "MED BAG") | (joined["p_container"] == "MED BOX") |
        (joined["p_container"] == "MED PKG") | (joined["p_container"] == "MED PACK")
    ).astype("int32") * md_grp
    lg_mask = (
        (joined["p_container"] == "LG CASE") | (joined["p_container"] == "LG BOX") |
        (joined["p_container"] == "LG PACK") | (joined["p_container"] == "LG PKG")
    ).astype("int32") * lg_grp
    container_grp = sm_mask + md_mask + lg_mask

    # Tensorize
    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(p_brand_id.to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(container_grp.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    # Launch fused CUDA kernel
    res = _EXT.q19_launcher(
        l_quantity_t, l_extendedprice_t, l_discount_t,
        p_brand_id_t, p_container_grp_t, p_size_t,
        brand12_id, brand23_id, brand34_id, sm_grp, md_grp, lg_grp
    )
    return float(res)
