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
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ quantity,
    const int8_t* __restrict__ part_match,
    double* __restrict__ out,
    int64_t n
) {
    __shared__ double block_rev[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (idx < n) {
        int8_t pm = part_match[idx];
        double qty = quantity[idx];
        bool keep = false;
        
        if (pm == 1) {
            keep = (qty >= 1.0 && qty <= 11.0);
        } else if (pm == 2) {
            keep = (qty >= 10.0 && qty <= 20.0);
        } else if (pm == 3) {
            keep = (qty >= 20.0 && qty <= 30.0);
        }
        
        if (keep) {
            my_rev = price[idx] * (1.0 - discount[idx]);
        }
    }
    
    block_rev[threadIdx.x] = my_rev;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_rev[threadIdx.x] += block_rev[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(out, block_rev[0]);
    }
}

double q19_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor quantity,
    torch::Tensor part_match
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64);
    TORCH_CHECK(part_match.scalar_type() == torch::kInt8);

    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();
    if (!part_match.is_cuda()) part_match = part_match.cuda();

    price = price.contiguous();
    discount = discount.contiguous();
    quantity = quantity.contiguous();
    part_match = part_match.contiguous();

    int64_t n = price.numel();
    if (n == 0) return 0.0;
    
    auto out = torch::zeros({1}, price.options());
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q19_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        quantity.data_ptr<double>(),
        part_match.data_ptr<int8_t>(),
        out.data_ptr<double>(),
        n
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
    Description: 
        Evaluates the complex disjunction by pre-computing valid conditions in cuDF for the `part` table, 
        filtering the `lineitem` table on shared ship properties, performing an inner join to only keep matched rows,
        and finally applying a single fused CUDA kernel. This minimizes memory overhead, limits intermediate sizes 
        before join, and aggregates the remaining filter (quantity) + revenue math + global sum natively on GPU.
    Params:
        brand1, brand2, brand3 (str): the three brand literals.
        shipinstruct (str): ship instruction literal.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # 1. Pre-filter lineitem on shipmode and shipinstruct
    l_sm = df_lineitem["l_shipmode"]
    l_si = df_lineitem["l_shipinstruct"]
    li_mask = l_sm.isin(["AIR", "AIR REG"]) & (l_si == shipinstruct)
    # Project only required columns early to minimize join width
    li = df_lineitem.loc[li_mask, ["l_partkey", "l_quantity", "l_extendedprice", "l_discount"]]

    # 2. Pre-filter part on brand combinations
    part_mask = df_part["p_brand"].isin([brand1, brand2, brand3])
    pt = df_part.loc[part_mask, ["p_partkey", "p_brand", "p_size", "p_container"]]
    
    brand = pt["p_brand"]
    size = pt["p_size"]
    cont = pt["p_container"]

    c1 = (brand == brand1) & (size >= 1) & (size <= 5) & cont.isin(['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'])
    c2 = (brand == brand2) & (size >= 1) & (size <= 10) & cont.isin(['MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'])
    c3 = (brand == brand3) & (size >= 1) & (size <= 15) & cont.isin(['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'])
    
    # Encode matched condition ID into an int8 series
    part_match = cudf.Series(0, index=pt.index, dtype="int8")
    part_match[c1] = 1
    part_match[c2] = 2
    part_match[c3] = 3
    
    pt["part_match"] = part_match
    # Keep only parts that match at least one sub-condition
    pt = pt.loc[part_match > 0, ["p_partkey", "part_match"]]

    # 3. Inner join on matching parts and lineitems
    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey", how="inner")
    
    if len(joined) == 0:
        return 0.0

    # 4. Map to Torch tensors and launch fused evaluate-and-sum kernel
    price = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    quantity = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    pm = torch.as_tensor(joined["part_match"].to_cupy(), dtype=torch.int8)

    res = _EXT.q19_launcher(price, discount, quantity, pm)
    return float(res)
