import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q19_kernel(
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ p_brand_id,
    const int32_t* __restrict__ p_container_grp,
    const int32_t* __restrict__ p_size,
    double* __restrict__ out,
    int64_t n,
    int32_t brand12_id,
    int32_t brand23_id,
    int32_t brand34_id,
    int32_t sm_grp,
    int32_t md_grp,
    int32_t lg_grp,
    double q1_lo, double q1_hi,
    double q2_lo, double q2_hi,
    double q3_lo, double q3_hi,
    int32_t s1_lo, int32_t s1_hi,
    int32_t s2_lo, int32_t s2_hi,
    int32_t s3_lo, int32_t s3_hi
) {
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0.0;
    if (idx < n) {
        double qty = l_quantity[idx];
        double price = l_extendedprice[idx];
        double disc = l_discount[idx];
        int32_t brand = p_brand_id[idx];
        int32_t cont = p_container_grp[idx];
        int32_t sz = p_size[idx];
        bool keep = false;
        if (brand == brand12_id && cont == sm_grp && qty >= q1_lo && qty <= q1_hi && sz >= s1_lo && sz <= s1_hi) keep = true;
        else if (brand == brand23_id && cont == md_grp && qty >= q2_lo && qty <= q2_hi && sz >= s2_lo && sz <= s2_hi) keep = true;
        else if (brand == brand34_id && cont == lg_grp && qty >= q3_lo && qty <= q3_hi && sz >= s3_lo && sz <= s3_hi) keep = true;
        if (keep) my_sum = price * (1.0 - disc);
    }
    block_sum[threadIdx.x] = my_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, block_sum[0]);
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
    int32_t lg_grp,
    double q1_lo, double q1_hi,
    double q2_lo, double q2_hi,
    double q3_lo, double q3_hi,
    int32_t s1_lo, int32_t s1_hi,
    int32_t s2_lo, int32_t s2_hi,
    int32_t s3_lo, int32_t s3_hi
) {
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(p_brand_id.scalar_type() == torch::kInt32);
    TORCH_CHECK(p_container_grp.scalar_type() == torch::kInt32);
    TORCH_CHECK(p_size.scalar_type() == torch::kInt32);
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
        out.data_ptr<double>(),
        n,
        brand12_id, brand23_id, brand34_id,
        sm_grp, md_grp, lg_grp,
        q1_lo, q1_hi, q2_lo, q2_hi, q3_lo, q3_hi,
        s1_lo, s1_hi, s2_lo, s2_hi, s3_lo, s3_hi
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
    Description: TPC-H Q19 - fused CUDA kernel for filter/project/aggregate in one pass.
    Params:
        brand1, brand2, brand3 (str): the three brand literals.
        shipinstruct (str): ship instruction literal.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]

    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part[part_mask]

    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    brand12_id, brand23_id, brand34_id = 12, 23, 34
    p_brand_col = joined["p_brand"]
    b12 = (p_brand_col == brand1).astype("int32") * brand12_id
    b23 = (p_brand_col == brand2).astype("int32") * brand23_id
    b34 = (p_brand_col == brand3).astype("int32") * brand34_id
    brand_encoded = b12 + b23 + b34

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

    res = _EXT.q19_launcher(
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
        1.0, 11.0,
        10.0, 20.0,
        20.0, 30.0,
        1, 5,
        1, 10,
        1, 15
    )
    return float(res)
