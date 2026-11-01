import os
import tempfile
from typing import Any

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#define BLOCK_SIZE 256
#define MAX_RF 16
#define MAX_LS 16
#define MAX_GROUPS (MAX_RF * MAX_LS)

__global__ void q1_aggregate_kernel(
    const int32_t* __restrict__ rf_ptr,
    const int32_t* __restrict__ ls_ptr,
    const double* __restrict__ qty_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    const double* __restrict__ tax_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t shipdate_hi_days,
    double* __restrict__ g_sum_qty,
    double* __restrict__ g_sum_base_price,
    double* __restrict__ g_sum_disc_price,
    double* __restrict__ g_sum_charge,
    double* __restrict__ g_sum_disc,
    int64_t* __restrict__ g_count,
    int64_t n_elements
) {
    __shared__ double s_sum_qty[MAX_GROUPS];
    __shared__ double s_sum_base_price[MAX_GROUPS];
    __shared__ double s_sum_disc_price[MAX_GROUPS];
    __shared__ double s_sum_charge[MAX_GROUPS];
    __shared__ double s_sum_disc[MAX_GROUPS];
    __shared__ unsigned long long s_count[MAX_GROUPS];

    const int tid = threadIdx.x;

    if (tid < MAX_GROUPS) {
        s_sum_qty[tid] = 0.0;
        s_sum_base_price[tid] = 0.0;
        s_sum_disc_price[tid] = 0.0;
        s_sum_charge[tid] = 0.0;
        s_sum_disc[tid] = 0.0;
        s_count[tid] = 0ULL;
    }
    __syncthreads();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    if (idx < n_elements) {
        const int32_t sd = shipdate_ptr[idx];
        if (sd <= shipdate_hi_days) {
            const int32_t rf = rf_ptr[idx];
            const int32_t ls = ls_ptr[idx];
            if (rf >= 0 && rf < MAX_RF && ls >= 0 && ls < MAX_LS) {
                const int g = rf * MAX_LS + ls;

                const double qty = qty_ptr[idx];
                const double price = price_ptr[idx];
                const double disc = disc_ptr[idx];
                const double tax = tax_ptr[idx];
                const double disc_price = price * (1.0 - disc);
                const double charge = disc_price * (1.0 + tax);

                atomicAdd(&s_sum_qty[g], qty);
                atomicAdd(&s_sum_base_price[g], price);
                atomicAdd(&s_sum_disc_price[g], disc_price);
                atomicAdd(&s_sum_charge[g], charge);
                atomicAdd(&s_sum_disc[g], disc);
                atomicAdd(&s_count[g], 1ULL);
            }
        }
    }

    __syncthreads();

    if (tid < MAX_GROUPS) {
        const unsigned long long c = s_count[tid];
        if (c != 0ULL) {
            atomicAdd(&g_sum_qty[tid], s_sum_qty[tid]);
            atomicAdd(&g_sum_base_price[tid], s_sum_base_price[tid]);
            atomicAdd(&g_sum_disc_price[tid], s_sum_disc_price[tid]);
            atomicAdd(&g_sum_charge[tid], s_sum_charge[tid]);
            atomicAdd(&g_sum_disc[tid], s_sum_disc[tid]);
            atomicAdd(reinterpret_cast<unsigned long long*>(&g_count[tid]), c);
        }
    }
}

__global__ void q1_finalize_kernel(
    const int64_t* __restrict__ group_ids,
    const double* __restrict__ g_sum_qty,
    const double* __restrict__ g_sum_base_price,
    const double* __restrict__ g_sum_disc_price,
    const double* __restrict__ g_sum_charge,
    const double* __restrict__ g_sum_disc,
    const int64_t* __restrict__ g_count,
    int64_t* __restrict__ out_rf,
    int64_t* __restrict__ out_ls,
    double* __restrict__ out_sum_qty,
    double* __restrict__ out_sum_base_price,
    double* __restrict__ out_sum_disc_price,
    double* __restrict__ out_sum_charge,
    double* __restrict__ out_avg_qty,
    double* __restrict__ out_avg_price,
    double* __restrict__ out_avg_disc,
    int64_t* __restrict__ out_count,
    int64_t n_groups
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_groups) {
        const int64_t gid = group_ids[i];
        const int64_t cnt = g_count[gid];
        const double cnt_d = static_cast<double>(cnt);

        out_rf[i] = gid / MAX_LS;
        out_ls[i] = gid % MAX_LS;

        const double sum_qty = g_sum_qty[gid];
        const double sum_base_price = g_sum_base_price[gid];
        const double sum_disc_price = g_sum_disc_price[gid];
        const double sum_charge = g_sum_charge[gid];
        const double sum_disc = g_sum_disc[gid];

        out_sum_qty[i] = sum_qty;
        out_sum_base_price[i] = sum_base_price;
        out_sum_disc_price[i] = sum_disc_price;
        out_sum_charge[i] = sum_charge;
        out_avg_qty[i] = sum_qty / cnt_d;
        out_avg_price[i] = sum_base_price / cnt_d;
        out_avg_disc[i] = sum_disc / cnt_d;
        out_count[i] = cnt;
    }
}

static void check_cuda_error(const char* where) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, where, ": ", cudaGetErrorString(err));
}

std::vector<torch::Tensor> q1_launcher(
    torch::Tensor l_returnflag_code,
    torch::Tensor l_linestatus_code,
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_tax,
    torch::Tensor l_shipdate_days,
    int32_t shipdate_hi_days
) {
    TORCH_CHECK(l_returnflag_code.is_cuda(), "l_returnflag_code must be CUDA");
    TORCH_CHECK(l_linestatus_code.is_cuda(), "l_linestatus_code must be CUDA");
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be CUDA");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be CUDA");
    TORCH_CHECK(l_discount.is_cuda(), "l_discount must be CUDA");
    TORCH_CHECK(l_tax.is_cuda(), "l_tax must be CUDA");
    TORCH_CHECK(l_shipdate_days.is_cuda(), "l_shipdate_days must be CUDA");

    TORCH_CHECK(l_returnflag_code.scalar_type() == torch::kInt32, "l_returnflag_code must be int32");
    TORCH_CHECK(l_linestatus_code.scalar_type() == torch::kInt32, "l_linestatus_code must be int32");
    TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32, "l_shipdate_days must be int32");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(l_tax.scalar_type() == torch::kFloat64, "l_tax must be float64");

    l_returnflag_code = l_returnflag_code.contiguous();
    l_linestatus_code = l_linestatus_code.contiguous();
    l_quantity = l_quantity.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    l_tax = l_tax.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();

    const int64_t n = l_quantity.numel();
    TORCH_CHECK(l_returnflag_code.numel() == n, "l_returnflag_code length mismatch");
    TORCH_CHECK(l_linestatus_code.numel() == n, "l_linestatus_code length mismatch");
    TORCH_CHECK(l_extendedprice.numel() == n, "l_extendedprice length mismatch");
    TORCH_CHECK(l_discount.numel() == n, "l_discount length mismatch");
    TORCH_CHECK(l_tax.numel() == n, "l_tax length mismatch");
    TORCH_CHECK(l_shipdate_days.numel() == n, "l_shipdate_days length mismatch");

    auto dbl_opts = l_quantity.options().dtype(torch::kFloat64);
    auto i64_opts = l_quantity.options().dtype(torch::kInt64);

    auto g_sum_qty = torch::zeros({MAX_GROUPS}, dbl_opts);
    auto g_sum_base_price = torch::zeros({MAX_GROUPS}, dbl_opts);
    auto g_sum_disc_price = torch::zeros({MAX_GROUPS}, dbl_opts);
    auto g_sum_charge = torch::zeros({MAX_GROUPS}, dbl_opts);
    auto g_sum_disc = torch::zeros({MAX_GROUPS}, dbl_opts);
    auto g_count = torch::zeros({MAX_GROUPS}, i64_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n > 0) {
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q1_aggregate_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            l_returnflag_code.data_ptr<int32_t>(),
            l_linestatus_code.data_ptr<int32_t>(),
            l_quantity.data_ptr<double>(),
            l_extendedprice.data_ptr<double>(),
            l_discount.data_ptr<double>(),
            l_tax.data_ptr<double>(),
            l_shipdate_days.data_ptr<int32_t>(),
            shipdate_hi_days,
            g_sum_qty.data_ptr<double>(),
            g_sum_base_price.data_ptr<double>(),
            g_sum_disc_price.data_ptr<double>(),
            g_sum_charge.data_ptr<double>(),
            g_sum_disc.data_ptr<double>(),
            g_count.data_ptr<int64_t>(),
            n
        );
        check_cuda_error("q1_aggregate_kernel launch failed");
    }

    auto count_cpu = g_count.to(torch::kCPU);
    const int64_t* count_ptr = count_cpu.data_ptr<int64_t>();

    int64_t n_groups = 0;
    for (int rf = 0; rf < MAX_RF; ++rf) {
        for (int ls = 0; ls < MAX_LS; ++ls) {
            const int gid = rf * MAX_LS + ls;
            if (count_ptr[gid] != 0) {
                ++n_groups;
            }
        }
    }

    auto out_rf = torch::empty({n_groups}, i64_opts);
    auto out_ls = torch::empty({n_groups}, i64_opts);
    auto out_sum_qty = torch::empty({n_groups}, dbl_opts);
    auto out_sum_base_price = torch::empty({n_groups}, dbl_opts);
    auto out_sum_disc_price = torch::empty({n_groups}, dbl_opts);
    auto out_sum_charge = torch::empty({n_groups}, dbl_opts);
    auto out_avg_qty = torch::empty({n_groups}, dbl_opts);
    auto out_avg_price = torch::empty({n_groups}, dbl_opts);
    auto out_avg_disc = torch::empty({n_groups}, dbl_opts);
    auto out_count = torch::empty({n_groups}, i64_opts);

    if (n_groups > 0) {
        auto gids_cpu = torch::empty({n_groups}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        int64_t* gids_ptr = gids_cpu.data_ptr<int64_t>();
        int64_t pos = 0;
        for (int rf = 0; rf < MAX_RF; ++rf) {
            for (int ls = 0; ls < MAX_LS; ++ls) {
                const int gid = rf * MAX_LS + ls;
                if (count_ptr[gid] != 0) {
                    gids_ptr[pos++] = static_cast<int64_t>(gid);
                }
            }
        }

        auto gids_dev = gids_cpu.to(l_quantity.device(), torch::kInt64, false, true);
        const int finalize_blocks = static_cast<int>((n_groups + 127) / 128);
        q1_finalize_kernel<<<finalize_blocks, 128, 0, stream>>>(
            gids_dev.data_ptr<int64_t>(),
            g_sum_qty.data_ptr<double>(),
            g_sum_base_price.data_ptr<double>(),
            g_sum_disc_price.data_ptr<double>(),
            g_sum_charge.data_ptr<double>(),
            g_sum_disc.data_ptr<double>(),
            g_count.data_ptr<int64_t>(),
            out_rf.data_ptr<int64_t>(),
            out_ls.data_ptr<int64_t>(),
            out_sum_qty.data_ptr<double>(),
            out_sum_base_price.data_ptr<double>(),
            out_sum_disc_price.data_ptr<double>(),
            out_sum_charge.data_ptr<double>(),
            out_avg_qty.data_ptr<double>(),
            out_avg_price.data_ptr<double>(),
            out_avg_disc.data_ptr<double>(),
            out_count.data_ptr<int64_t>(),
            n_groups
        );
        check_cuda_error("q1_finalize_kernel launch failed");
    }

    return {
        out_rf,
        out_ls,
        out_sum_qty,
        out_sum_base_price,
        out_sum_disc_price,
        out_sum_charge,
        out_avg_qty,
        out_avg_price,
        out_avg_disc,
        out_count
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q1_launcher", &q1_launcher, "TPC-H Q1 fused filter/project/group aggregate (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension from the in-module .cu source.

    The device code performs the memory-bound TPC-H Q1 filter, projection, and
    small-cardinality aggregation in one CUDA pass, avoiding the reference
    PyTorch hot path's materialized masks, filtered temporaries, unique, and
    scatter-add passes.
    """
    tmpdir = tempfile.mkdtemp(prefix="q1_cuda_")
    cu_path = os.path.join(tmpdir, "_q1_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q1_cuda_ext",
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """Run the numeric TPC-H Q1 hot path on the GPU with a fused CUDA kernel.

    CUDA scans the aligned 1D tensors once, applies the ship-date predicate,
    computes discounted price and charge in registers, and aggregates into the
    small fixed returnflag/linestatus code space using per-block shared-memory
    accumulators followed by one global atomic update per group per block.  This
    is faster than the eager PyTorch reference for this memory-bound query
    because it avoids filtered tensor materialization, `torch.unique`, inverse
    indices, and multiple scatter-add passes over large intermediate arrays.
    """
    result = _EXT.q1_launcher(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        int(shipdate_hi_days),
    )
    return (
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
        result[6],
        result[7],
        result[8],
        result[9],
    )


def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
    Description: TPC-H Q1 pricing summary report aggregated by return flag and line status.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]

    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode string columns l_returnflag, l_linestatus to integer codes (single-char strings).
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    # Use categorical codes for encoding
    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

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
        rf_out, ls_out, sum_qty, sum_base_price, sum_disc_price,
        sum_charge, avg_qty, avg_price, avg_disc, count_order,
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
    )

    # Decode category codes back to strings
    rf_out_host = rf_out.cpu().tolist()
    ls_out_host = ls_out.cpu().tolist()
    rf_strs = [rf_categories[i] for i in rf_out_host]
    ls_strs = [ls_categories[i] for i in ls_out_host]

    return cudf.DataFrame({
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
    })
