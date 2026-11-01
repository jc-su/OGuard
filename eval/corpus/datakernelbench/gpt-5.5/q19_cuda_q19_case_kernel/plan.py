import os
import tempfile
from typing import Any

import cudf
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#define BLOCK_SIZE 1024

static int compute_blocks(const int64_t n_elements) {
    if (n_elements <= 0) {
        return 0;
    }
    const int needed = static_cast<int>((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int cap = std::max(1, prop.multiProcessorCount * 8);
    return std::min(needed, cap);
}

__global__ void q19_case_kernel(
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ qcase_ptr,
    double* __restrict__ out_ptr,
    const int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    double my_rev = 0.0;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_elements;
         idx += stride) {
        const int32_t qcase = qcase_ptr[idx];
        const double qty = quantity_ptr[idx];

        const bool keep =
            ((qcase == 1) && (qty >= 1.0)  && (qty <= 11.0)) ||
            ((qcase == 2) && (qty >= 10.0) && (qty <= 20.0)) ||
            ((qcase == 3) && (qty >= 20.0) && (qty <= 30.0));

        if (keep) {
            const double price = price_ptr[idx];
            const double disc = discount_ptr[idx];
            my_rev += price * (1.0 - disc);
        }
    }

    block_revenue[tid] = my_rev;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_revenue[tid] += block_revenue[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}

__global__ void q19_full_kernel(
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ brand_ptr,
    const int32_t* __restrict__ container_ptr,
    const int32_t* __restrict__ size_ptr,
    const int32_t brand12_id,
    const int32_t brand23_id,
    const int32_t brand34_id,
    const int32_t sm_grp,
    const int32_t md_grp,
    const int32_t lg_grp,
    double* __restrict__ out_ptr,
    const int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    double my_rev = 0.0;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_elements;
         idx += stride) {
        const double qty = quantity_ptr[idx];
        const int32_t brand = brand_ptr[idx];
        const int32_t cont = container_ptr[idx];
        const int32_t psz = size_ptr[idx];

        const bool cond1 =
            (brand == brand12_id) &&
            (cont == sm_grp) &&
            (qty >= 1.0) &&
            (qty <= 11.0) &&
            (psz >= 1) &&
            (psz <= 5);

        const bool cond2 =
            (brand == brand23_id) &&
            (cont == md_grp) &&
            (qty >= 10.0) &&
            (qty <= 20.0) &&
            (psz >= 1) &&
            (psz <= 10);

        const bool cond3 =
            (brand == brand34_id) &&
            (cont == lg_grp) &&
            (qty >= 20.0) &&
            (qty <= 30.0) &&
            (psz >= 1) &&
            (psz <= 15);

        if (cond1 || cond2 || cond3) {
            const double price = price_ptr[idx];
            const double disc = discount_ptr[idx];
            my_rev += price * (1.0 - disc);
        }
    }

    block_revenue[tid] = my_rev;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_revenue[tid] += block_revenue[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}

double q19_case_launcher(
    torch::Tensor quantity,
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor qcase
) {
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(qcase.scalar_type() == torch::kInt32, "qcase must be int32");

    if (!quantity.is_cuda()) quantity = quantity.to(torch::kCUDA);
    if (!price.is_cuda()) price = price.to(torch::kCUDA);
    if (!discount.is_cuda()) discount = discount.to(torch::kCUDA);
    if (!qcase.is_cuda()) qcase = qcase.to(torch::kCUDA);

    quantity = quantity.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();
    qcase = qcase.contiguous();

    const int64_t n = quantity.numel();
    TORCH_CHECK(price.numel() == n, "price length mismatch");
    TORCH_CHECK(discount.numel() == n, "discount length mismatch");
    TORCH_CHECK(qcase.numel() == n, "qcase length mismatch");

    if (n == 0) {
        return 0.0;
    }

    auto out = torch::zeros({1}, quantity.options().dtype(torch::kFloat64));
    const int blocks = compute_blocks(n);

    q19_case_kernel<<<blocks, BLOCK_SIZE>>>(
        quantity.data_ptr<double>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        qcase.data_ptr<int32_t>(),
        out.data_ptr<double>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out.item<double>();
}

double q19_full_launcher(
    torch::Tensor quantity,
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor brand,
    torch::Tensor container,
    torch::Tensor size,
    const int32_t brand12_id,
    const int32_t brand23_id,
    const int32_t brand34_id,
    const int32_t sm_grp,
    const int32_t md_grp,
    const int32_t lg_grp
) {
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(brand.scalar_type() == torch::kInt32, "brand must be int32");
    TORCH_CHECK(container.scalar_type() == torch::kInt32, "container must be int32");
    TORCH_CHECK(size.scalar_type() == torch::kInt32, "size must be int32");

    if (!quantity.is_cuda()) quantity = quantity.to(torch::kCUDA);
    if (!price.is_cuda()) price = price.to(torch::kCUDA);
    if (!discount.is_cuda()) discount = discount.to(torch::kCUDA);
    if (!brand.is_cuda()) brand = brand.to(torch::kCUDA);
    if (!container.is_cuda()) container = container.to(torch::kCUDA);
    if (!size.is_cuda()) size = size.to(torch::kCUDA);

    quantity = quantity.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();
    brand = brand.contiguous();
    container = container.contiguous();
    size = size.contiguous();

    const int64_t n = quantity.numel();
    TORCH_CHECK(price.numel() == n, "price length mismatch");
    TORCH_CHECK(discount.numel() == n, "discount length mismatch");
    TORCH_CHECK(brand.numel() == n, "brand length mismatch");
    TORCH_CHECK(container.numel() == n, "container length mismatch");
    TORCH_CHECK(size.numel() == n, "size length mismatch");

    if (n == 0) {
        return 0.0;
    }

    auto out = torch::zeros({1}, quantity.options().dtype(torch::kFloat64));
    const int blocks = compute_blocks(n);

    q19_full_kernel<<<blocks, BLOCK_SIZE>>>(
        quantity.data_ptr<double>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        brand.data_ptr<int32_t>(),
        container.data_ptr<int32_t>(),
        size.data_ptr<int32_t>(),
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
        out.data_ptr<double>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q19_case_launcher", &q19_case_launcher, "TPC-H Q19 optimized case-filtered revenue (CUDA)");
    m.def("q19_full_launcher", &q19_full_launcher, "TPC-H Q19 full predicate revenue (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension once at import; kernels perform the GPU hot-path scan/reduction."""
    tmpdir = tempfile.mkdtemp(prefix="q19_cuda_")
    cu_path = os.path.join(tmpdir, "_q19_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q19_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _to_float64_tensor(series: cudf.Series) -> torch.Tensor:
    """Convert a cuDF numeric column to a CUDA torch float64 tensor without host materialization."""
    return torch.as_tensor(series.to_cupy(), dtype=torch.float64)


def _to_int32_tensor(series: cudf.Series) -> torch.Tensor:
    """Convert a cuDF numeric column to a CUDA torch int32 tensor without host materialization."""
    return torch.as_tensor(series.to_cupy(), dtype=torch.int32)


def _launch_q19_case(
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    qcase: torch.Tensor,
) -> torch.Tensor:
    """
    Run the optimized CUDA Q19 hot path.

    The CUDA kernel performs a fused coalesced scan over joined rows, applies the remaining
    quantity predicate in registers, conditionally loads price/discount only for kept rows,
    and reduces with one atomicAdd per block.
    """
    rev = _EXT.q19_case_launcher(l_quantity, l_extendedprice, l_discount, qcase)
    return torch.tensor(rev, dtype=torch.float64, device=l_quantity.device)


def _launch_q19_full(
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
    Run the exact reference-style CUDA hot path.

    This fallback keeps reference brand/container encodings and fuses filter/project/sum into
    one CUDA scan, avoiding eager PyTorch boolean masks and temporaries.
    """
    rev = _EXT.q19_full_launcher(
        l_quantity,
        l_extendedprice,
        l_discount,
        p_brand_id,
        p_container_grp,
        p_size,
        int(brand12_id),
        int(brand23_id),
        int(brand34_id),
        int(sm_grp),
        int(md_grp),
        int(lg_grp),
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
    Reference-compatible core implemented in CUDA.

    It replaces eager PyTorch mask construction with one fused CUDA kernel, which is faster for
    this memory-bound scan because it avoids multiple temporary boolean/revenue tensors and
    performs a block reduction before global atomics.
    """
    return _launch_q19_full(
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


def _run_query_exact_reference_semantics(
    tables: dict[str, cudf.DataFrame],
    brand1: str,
    brand2: str,
    brand3: str,
    shipinstruct: str,
) -> float:
    """
    cuDF performs string filtering and the part/lineitem join; CUDA performs the fused numeric
    predicate/project/aggregate with the same brand encoding behavior as the reference.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
        & (df_lineitem["l_quantity"] >= 1.0)
        & (df_lineitem["l_quantity"] <= 30.0)
    )
    li = df_lineitem.loc[
        li_mask,
        ["l_partkey", "l_quantity", "l_extendedprice", "l_discount"],
    ]

    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part.loc[
        part_mask,
        ["p_partkey", "p_brand", "p_container", "p_size"],
    ]

    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    p_brand_col = joined["p_brand"]
    brand_encoded = (
        ((p_brand_col == brand1).astype("int32") * 12)
        + ((p_brand_col == brand2).astype("int32") * 23)
        + ((p_brand_col == brand3).astype("int32") * 34)
    ).astype("int32")

    cont = joined["p_container"]
    container_grp = (
        (
            (
                (cont == "SM CASE")
                | (cont == "SM BOX")
                | (cont == "SM PACK")
                | (cont == "SM PKG")
            ).astype("int32")
            * 1
        )
        + (
            (
                (cont == "MED BAG")
                | (cont == "MED BOX")
                | (cont == "MED PKG")
                | (cont == "MED PACK")
            ).astype("int32")
            * 2
        )
        + (
            (
                (cont == "LG CASE")
                | (cont == "LG BOX")
                | (cont == "LG PACK")
                | (cont == "LG PKG")
            ).astype("int32")
            * 3
        )
    ).astype("int32")

    l_quantity_t = _to_float64_tensor(joined["l_quantity"])
    l_extendedprice_t = _to_float64_tensor(joined["l_extendedprice"])
    l_discount_t = _to_float64_tensor(joined["l_discount"])
    p_brand_id_t = _to_int32_tensor(brand_encoded)
    p_container_grp_t = _to_int32_tensor(container_grp)
    p_size_t = _to_int32_tensor(joined["p_size"].astype("int32"))

    res = _query_core(
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        p_brand_id_t,
        p_container_grp_t,
        p_size_t,
        12,
        23,
        34,
        1,
        2,
        3,
    )
    return float(res.item())


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
        WHERE (p_partkey = l_partkey
               AND p_brand = 'Brand#12'
               AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
               AND l_quantity >= 1 AND l_quantity <= 11
               AND p_size BETWEEN 1 AND 5
               AND l_shipmode IN ('AIR', 'AIR REG')
               AND l_shipinstruct = 'DELIVER IN PERSON')
           OR (p_partkey = l_partkey
               AND p_brand = 'Brand#23'
               AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
               AND l_quantity >= 10 AND l_quantity <= 20
               AND p_size BETWEEN 1 AND 10
               AND l_shipmode IN ('AIR', 'AIR REG')
               AND l_shipinstruct = 'DELIVER IN PERSON')
           OR (p_partkey = l_partkey
               AND p_brand = 'Brand#34'
               AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
               AND l_quantity >= 20 AND l_quantity <= 30
               AND p_size BETWEEN 1 AND 15
               AND l_shipmode IN ('AIR', 'AIR REG')
               AND l_shipinstruct = 'DELIVER IN PERSON').

    Description:
        TPC-H Q19 discounted revenue. cuDF handles string predicates and the equi-join.
        For the common distinct-brand case, cuDF also pre-applies brand/container/size
        disjuncts on the small part side and stores a compact integer case id. The CUDA hot
        path then performs one fused memory-bound scan over joined numeric columns:
        remaining quantity filter, conditional price/discount load, projection, and block
        reduction with one global atomic per block. This avoids the reference's eager
        PyTorch boolean masks, revenue temporary, and multiple HBM passes.

    Params:
        tables: Dictionary containing cuDF DataFrames, including "lineitem" and "part".
        brand1, brand2, brand3: Brand literals for the three Q19 disjuncts.
        shipinstruct: Required l_shipinstruct literal.
    """
    if len({brand1, brand2, brand3}) != 3:
        return _run_query_exact_reference_semantics(tables, brand1, brand2, brand3, shipinstruct)

    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
        & (df_lineitem["l_quantity"] >= 1.0)
        & (df_lineitem["l_quantity"] <= 30.0)
    )
    li = df_lineitem.loc[
        li_mask,
        ["l_partkey", "l_quantity", "l_extendedprice", "l_discount"],
    ]

    p_size = df_part["p_size"]
    cont = df_part["p_container"]
    sm_cont = (
        (cont == "SM CASE")
        | (cont == "SM BOX")
        | (cont == "SM PACK")
        | (cont == "SM PKG")
    )
    md_cont = (
        (cont == "MED BAG")
        | (cont == "MED BOX")
        | (cont == "MED PKG")
        | (cont == "MED PACK")
    )
    lg_cont = (
        (cont == "LG CASE")
        | (cont == "LG BOX")
        | (cont == "LG PACK")
        | (cont == "LG PKG")
    )

    mask1 = (df_part["p_brand"] == brand1) & sm_cont & (p_size >= 1) & (p_size <= 5)
    mask2 = (df_part["p_brand"] == brand2) & md_cont & (p_size >= 1) & (p_size <= 10)
    mask3 = (df_part["p_brand"] == brand3) & lg_cont & (p_size >= 1) & (p_size <= 15)

    pt1 = df_part.loc[mask1, ["p_partkey"]].copy()
    pt1["qcase"] = 1
    pt2 = df_part.loc[mask2, ["p_partkey"]].copy()
    pt2["qcase"] = 2
    pt3 = df_part.loc[mask3, ["p_partkey"]].copy()
    pt3["qcase"] = 3

    pt = cudf.concat([pt1, pt2, pt3], ignore_index=True)
    pt["qcase"] = pt["qcase"].astype("int32")

    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    l_quantity_t = _to_float64_tensor(joined["l_quantity"])
    l_extendedprice_t = _to_float64_tensor(joined["l_extendedprice"])
    l_discount_t = _to_float64_tensor(joined["l_discount"])
    qcase_t = _to_int32_tensor(joined["qcase"])

    res = _launch_q19_case(
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        qcase_t,
    )
    return float(res.item())
