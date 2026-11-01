import os
import tempfile
from typing import Any

import cudf
import pandas as pd
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 512
#define MAX_GROUPS 64
#define MAX_BLOCKS 8192

#define ADD_ROW_TO_GROUP(G) do {                                      \
    q##G += qty;                                                       \
    p##G += price;                                                     \
    double _dp = price * (1.0 - disc);                                 \
    dp##G += _dp;                                                      \
    ch##G += _dp * (1.0 + tax);                                        \
    d##G += disc;                                                      \
    c##G += 1LL;                                                       \
} while (0)

#define REDUCE_AND_ATOMIC(G, VQ, VP, VDP, VCH, VD, VC) do {            \
    sh_qty[threadIdx.x] = (VQ);                                        \
    sh_price[threadIdx.x] = (VP);                                      \
    sh_disc_price[threadIdx.x] = (VDP);                                \
    sh_charge[threadIdx.x] = (VCH);                                    \
    sh_disc[threadIdx.x] = (VD);                                       \
    sh_count[threadIdx.x] = (VC);                                      \
    __syncthreads();                                                   \
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {                     \
        if (threadIdx.x < s) {                                         \
            sh_qty[threadIdx.x] += sh_qty[threadIdx.x + s];            \
            sh_price[threadIdx.x] += sh_price[threadIdx.x + s];        \
            sh_disc_price[threadIdx.x] += sh_disc_price[threadIdx.x + s]; \
            sh_charge[threadIdx.x] += sh_charge[threadIdx.x + s];      \
            sh_disc[threadIdx.x] += sh_disc[threadIdx.x + s];          \
            sh_count[threadIdx.x] += sh_count[threadIdx.x + s];        \
        }                                                             \
        __syncthreads();                                               \
    }                                                                 \
    if (threadIdx.x == 0 && (G) < n_groups) {                          \
        atomicAdd(out_qty + (G), sh_qty[0]);                           \
        atomicAdd(out_price + (G), sh_price[0]);                       \
        atomicAdd(out_disc_price + (G), sh_disc_price[0]);             \
        atomicAdd(out_charge + (G), sh_charge[0]);                     \
        atomicAdd(out_disc + (G), sh_disc[0]);                         \
        atomicAdd(reinterpret_cast<unsigned long long*>(out_count + (G)), \
                  static_cast<unsigned long long>(sh_count[0]));       \
    }                                                                 \
    __syncthreads();                                                   \
} while (0)

__global__ void q1_kernel6(
    const int32_t* __restrict__ returnflag,
    const int32_t* __restrict__ linestatus,
    const double* __restrict__ quantity,
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    const double* __restrict__ tax,
    const int32_t* __restrict__ shipdate_days,
    int32_t shipdate_hi_days,
    int32_t n_ls,
    int32_t n_groups,
    int64_t n,
    double* __restrict__ out_qty,
    double* __restrict__ out_price,
    double* __restrict__ out_disc_price,
    double* __restrict__ out_charge,
    double* __restrict__ out_disc,
    int64_t* __restrict__ out_count
) {
    __shared__ double sh_qty[BLOCK_SIZE];
    __shared__ double sh_price[BLOCK_SIZE];
    __shared__ double sh_disc_price[BLOCK_SIZE];
    __shared__ double sh_charge[BLOCK_SIZE];
    __shared__ double sh_disc[BLOCK_SIZE];
    __shared__ long long sh_count[BLOCK_SIZE];

    double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0, q4 = 0.0, q5 = 0.0;
    double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0, p5 = 0.0;
    double dp0 = 0.0, dp1 = 0.0, dp2 = 0.0, dp3 = 0.0, dp4 = 0.0, dp5 = 0.0;
    double ch0 = 0.0, ch1 = 0.0, ch2 = 0.0, ch3 = 0.0, ch4 = 0.0, ch5 = 0.0;
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0, d4 = 0.0, d5 = 0.0;
    long long c0 = 0LL, c1 = 0LL, c2 = 0LL, c3 = 0LL, c4 = 0LL, c5 = 0LL;

    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK_SIZE;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         idx < n;
         idx += stride) {
        int32_t sd = shipdate_days[idx];
        if (sd <= shipdate_hi_days) {
            int32_t rf = returnflag[idx];
            int32_t ls = linestatus[idx];
            int32_t gid = rf * n_ls + ls;
            if (rf >= 0 && ls >= 0 && gid >= 0 && gid < n_groups) {
                double qty = quantity[idx];
                double price = extendedprice[idx];
                double disc = discount[idx];
                double tax_v = tax[idx];
                tax = tax;  // keep parameter name live for older nvcc diagnostics
                double tax_local = tax_v;
                #define tax tax_local
                switch (gid) {
                    case 0: ADD_ROW_TO_GROUP(0); break;
                    case 1: ADD_ROW_TO_GROUP(1); break;
                    case 2: ADD_ROW_TO_GROUP(2); break;
                    case 3: ADD_ROW_TO_GROUP(3); break;
                    case 4: ADD_ROW_TO_GROUP(4); break;
                    case 5: ADD_ROW_TO_GROUP(5); break;
                    default: break;
                }
                #undef tax
            }
        }
    }

    REDUCE_AND_ATOMIC(0, q0, p0, dp0, ch0, d0, c0);
    REDUCE_AND_ATOMIC(1, q1, p1, dp1, ch1, d1, c1);
    REDUCE_AND_ATOMIC(2, q2, p2, dp2, ch2, d2, c2);
    REDUCE_AND_ATOMIC(3, q3, p3, dp3, ch3, d3, c3);
    REDUCE_AND_ATOMIC(4, q4, p4, dp4, ch4, d4, c4);
    REDUCE_AND_ATOMIC(5, q5, p5, dp5, ch5, d5, c5);
}

__global__ void q1_kernel_generic(
    const int32_t* __restrict__ returnflag,
    const int32_t* __restrict__ linestatus,
    const double* __restrict__ quantity,
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    const double* __restrict__ tax,
    const int32_t* __restrict__ shipdate_days,
    int32_t shipdate_hi_days,
    int32_t n_ls,
    int32_t n_groups,
    int64_t n,
    double* __restrict__ out_qty,
    double* __restrict__ out_price,
    double* __restrict__ out_disc_price,
    double* __restrict__ out_charge,
    double* __restrict__ out_disc,
    int64_t* __restrict__ out_count
) {
    __shared__ double sh_qty[MAX_GROUPS];
    __shared__ double sh_price[MAX_GROUPS];
    __shared__ double sh_disc_price[MAX_GROUPS];
    __shared__ double sh_charge[MAX_GROUPS];
    __shared__ double sh_disc[MAX_GROUPS];
    __shared__ unsigned long long sh_count[MAX_GROUPS];

    for (int g = threadIdx.x; g < n_groups; g += BLOCK_SIZE) {
        sh_qty[g] = 0.0;
        sh_price[g] = 0.0;
        sh_disc_price[g] = 0.0;
        sh_charge[g] = 0.0;
        sh_disc[g] = 0.0;
        sh_count[g] = 0ULL;
    }
    __syncthreads();

    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK_SIZE;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         idx < n;
         idx += stride) {
        int32_t sd = shipdate_days[idx];
        if (sd <= shipdate_hi_days) {
            int32_t rf = returnflag[idx];
            int32_t ls = linestatus[idx];
            int32_t gid = rf * n_ls + ls;
            if (rf >= 0 && ls >= 0 && gid >= 0 && gid < n_groups) {
                double qty = quantity[idx];
                double price = extendedprice[idx];
                double disc = discount[idx];
                double tax_v = tax[idx];
                double disc_price = price * (1.0 - disc);
                atomicAdd(&sh_qty[gid], qty);
                atomicAdd(&sh_price[gid], price);
                atomicAdd(&sh_disc_price[gid], disc_price);
                atomicAdd(&sh_charge[gid], disc_price * (1.0 + tax_v));
                atomicAdd(&sh_disc[gid], disc);
                atomicAdd(&sh_count[gid], 1ULL);
            }
        }
    }
    __syncthreads();

    for (int g = threadIdx.x; g < n_groups; g += BLOCK_SIZE) {
        if (sh_count[g] != 0ULL) {
            atomicAdd(out_qty + g, sh_qty[g]);
            atomicAdd(out_price + g, sh_price[g]);
            atomicAdd(out_disc_price + g, sh_disc_price[g]);
            atomicAdd(out_charge + g, sh_charge[g]);
            atomicAdd(out_disc + g, sh_disc[g]);
            atomicAdd(reinterpret_cast<unsigned long long*>(out_count + g), sh_count[g]);
        }
    }
}

std::vector<torch::Tensor> q1_launcher(
    torch::Tensor returnflag,
    torch::Tensor linestatus,
    torch::Tensor quantity,
    torch::Tensor extendedprice,
    torch::Tensor discount,
    torch::Tensor tax,
    torch::Tensor shipdate_days,
    int32_t shipdate_hi_days,
    int64_t n_rf,
    int64_t n_ls
) {
    if (!returnflag.is_cuda()) returnflag = returnflag.cuda();
    if (!linestatus.is_cuda()) linestatus = linestatus.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();
    if (!extendedprice.is_cuda()) extendedprice = extendedprice.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!tax.is_cuda()) tax = tax.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();

    returnflag = returnflag.contiguous();
    linestatus = linestatus.contiguous();
    quantity = quantity.contiguous();
    extendedprice = extendedprice.contiguous();
    discount = discount.contiguous();
    tax = tax.contiguous();
    shipdate_days = shipdate_days.contiguous();

    TORCH_CHECK(returnflag.scalar_type() == torch::kInt32, "returnflag codes must be int32");
    TORCH_CHECK(linestatus.scalar_type() == torch::kInt32, "linestatus codes must be int32");
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32, "shipdate_days must be int32 epoch days");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(extendedprice.scalar_type() == torch::kFloat64, "extendedprice must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(tax.scalar_type() == torch::kFloat64, "tax must be float64");

    int64_t n = quantity.numel();
    TORCH_CHECK(returnflag.numel() == n && linestatus.numel() == n &&
                extendedprice.numel() == n && discount.numel() == n &&
                tax.numel() == n && shipdate_days.numel() == n,
                "all input tensors must have the same length");

    TORCH_CHECK(n_rf >= 0 && n_ls >= 0, "category counts must be non-negative");
    int64_t n_groups64 = n_rf * n_ls;
    TORCH_CHECK(n_groups64 <= MAX_GROUPS, "q1 CUDA kernel supports at most 64 category combinations");
    int32_t n_groups = static_cast<int32_t>(n_groups64);
    int32_t n_ls_i = static_cast<int32_t>(n_ls);

    auto double_opts = quantity.options().dtype(torch::kFloat64);
    auto count_opts = quantity.options().dtype(torch::kInt64);

    auto out_qty = torch::zeros({n_groups64}, double_opts);
    auto out_price = torch::zeros({n_groups64}, double_opts);
    auto out_disc_price = torch::zeros({n_groups64}, double_opts);
    auto out_charge = torch::zeros({n_groups64}, double_opts);
    auto out_disc = torch::zeros({n_groups64}, double_opts);
    auto out_count = torch::zeros({n_groups64}, count_opts);

    if (n == 0 || n_groups == 0) {
        return {out_qty, out_price, out_disc_price, out_charge, out_disc, out_count};
    }

    int64_t blocks64 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks = static_cast<int>(std::min<int64_t>(blocks64, MAX_BLOCKS));
    blocks = std::max(blocks, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n_groups <= 6) {
        q1_kernel6<<<blocks, BLOCK_SIZE, 0, stream>>>(
            returnflag.data_ptr<int32_t>(),
            linestatus.data_ptr<int32_t>(),
            quantity.data_ptr<double>(),
            extendedprice.data_ptr<double>(),
            discount.data_ptr<double>(),
            tax.data_ptr<double>(),
            shipdate_days.data_ptr<int32_t>(),
            shipdate_hi_days,
            n_ls_i,
            n_groups,
            n,
            out_qty.data_ptr<double>(),
            out_price.data_ptr<double>(),
            out_disc_price.data_ptr<double>(),
            out_charge.data_ptr<double>(),
            out_disc.data_ptr<double>(),
            out_count.data_ptr<int64_t>()
        );
    } else {
        q1_kernel_generic<<<blocks, BLOCK_SIZE, 0, stream>>>(
            returnflag.data_ptr<int32_t>(),
            linestatus.data_ptr<int32_t>(),
            quantity.data_ptr<double>(),
            extendedprice.data_ptr<double>(),
            discount.data_ptr<double>(),
            tax.data_ptr<double>(),
            shipdate_days.data_ptr<int32_t>(),
            shipdate_hi_days,
            n_ls_i,
            n_groups,
            n,
            out_qty.data_ptr<double>(),
            out_price.data_ptr<double>(),
            out_disc_price.data_ptr<double>(),
            out_charge.data_ptr<double>(),
            out_disc.data_ptr<double>(),
            out_count.data_ptr<int64_t>()
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_qty, out_price, out_disc_price, out_charge, out_disc, out_count};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q1_launcher", &q1_launcher, "TPC-H Q1 fused filter/project/group aggregate (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the embedded CUDA extension from a temporary .cu file for the Q1 hot path."""
    tmpdir = tempfile.mkdtemp(prefix="q1_cuda_full_")
    cu_path = os.path.join(tmpdir, "_q1_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q1_cuda_full_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )


_EXT: Any = _load_cuda()


def _query_core(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
    n_rf: int,
    n_ls: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the hot path in one CUDA extension call.

    The CUDA kernel performs the date filter, price/discount/tax projections, and fixed-small-cardinality
    GROUP BY aggregation in a fused scan over lineitem columns. This avoids PyTorch mask materialization,
    compaction, torch.unique, and multiple scatter_add passes; for this memory-bound TPC-H Q1 scan it
    reduces HBM traffic and intermediate tensors.
    """
    outs = _EXT.q1_launcher(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        int(shipdate_hi_days),
        int(n_rf),
        int(n_ls),
    )
    return outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]


def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL:
        SELECT
            l_returnflag,
            l_linestatus,
            sum(l_quantity) AS sum_qty,
            sum(l_extendedprice) AS sum_base_price,
            sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            avg(l_quantity) AS avg_qty,
            avg(l_extendedprice) AS avg_price,
            avg(l_discount) AS avg_disc,
            count(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= CAST('1998-09-02' AS date)
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus;

    Description:
        TPC-H Q1 pricing summary report. cuDF is used for string categorical encoding and final
        DataFrame construction. The bulk numeric work runs in a hand-written CUDA extension: one fused,
        coalesced pass filters rows by int32 epoch-day shipdate, computes discounted price and charge in
        registers, and aggregates into the small (returnflag, linestatus) group space with block-level
        reductions before global atomics. This is faster than the reference PyTorch plan because it avoids
        large boolean masks, filtered-column temporaries, torch.unique, and several scatter_add passes.

    Params:
        tables: dictionary containing the cuDF DataFrame "lineitem".
        shipdate_hi: inclusive upper-bound ship date as a string parseable by pandas.
    """
    df_lineitem = tables["lineitem"]
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    rf_codes = df_lineitem["l_returnflag"].astype("str").astype("category")
    ls_codes = df_lineitem["l_linestatus"].astype("str").astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()
    n_rf = len(rf_categories)
    n_ls = len(ls_categories)

    l_returnflag_code_t = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy())
    l_linestatus_code_t = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy())
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    (
        sum_qty_full,
        sum_base_price_full,
        sum_disc_price_full,
        sum_charge_full,
        sum_disc_full,
        count_order_full,
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
        n_rf,
        n_ls,
    )

    count_host = count_order_full.detach().cpu().tolist()
    nonzero_indices = [i for i, c in enumerate(count_host) if int(c) != 0]
    idx_t = torch.as_tensor(nonzero_indices, dtype=torch.int64, device=sum_qty_full.device)

    sum_qty = sum_qty_full.index_select(0, idx_t)
    sum_base_price = sum_base_price_full.index_select(0, idx_t)
    sum_disc_price = sum_disc_price_full.index_select(0, idx_t)
    sum_charge = sum_charge_full.index_select(0, idx_t)
    sum_disc = sum_disc_full.index_select(0, idx_t)
    count_order = count_order_full.index_select(0, idx_t)

    count_f = count_order.to(torch.float64)
    avg_qty = sum_qty / count_f
    avg_price = sum_base_price / count_f
    avg_disc = sum_disc / count_f

    rf_strs = [rf_categories[i // n_ls] for i in nonzero_indices]
    ls_strs = [ls_categories[i % n_ls] for i in nonzero_indices]

    return cudf.DataFrame(
        {
            "l_returnflag": cudf.Series(rf_strs),
            "l_linestatus": cudf.Series(ls_strs),
            "sum_qty": sum_qty,
            "sum_base_price": sum_base_price,
            "sum_disc_price": sum_disc_price,
            "sum_charge": sum_charge,
            "avg_qty": avg_qty,
            "avg_price": avg_price,
            "avg_disc": avg_disc,
            "count_order": count_order,
        }
    )
