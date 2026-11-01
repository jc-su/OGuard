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
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256
#define EMPTY_KEY 0xffffffffffffffffULL

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t make_pair_key_i64(int64_t partkey, int64_t suppkey, int64_t key_base) {
    return ((uint64_t)partkey) * ((uint64_t)key_base) + ((uint64_t)suppkey);
}

__global__ void build_ps_hash_kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    uint64_t* __restrict__ table_keys,
    int64_t* __restrict__ table_vals,
    int64_t nps,
    int64_t key_base,
    int64_t table_mask
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nps) {
        return;
    }

    uint64_t key = make_pair_key_i64(ps_partkey[idx], ps_suppkey[idx], key_base);
    uint64_t h = mix64(key) & (uint64_t)table_mask;

    while (true) {
        uint64_t old = atomicCAS(
            (unsigned long long*)&table_keys[h],
            (unsigned long long)EMPTY_KEY,
            (unsigned long long)key
        );
        if (old == EMPTY_KEY || old == key) {
            table_vals[h] = idx;
            return;
        }
        h = (h + 1ULL) & (uint64_t)table_mask;
    }
}

__global__ void scan_lineitem_accum_kernel(
    const int64_t* __restrict__ l_partkey,
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_quantity,
    const int32_t* __restrict__ l_shipdate_days,
    const uint64_t* __restrict__ table_keys,
    const int64_t* __restrict__ table_vals,
    double* __restrict__ ps_sum_qty,
    int64_t nli,
    int64_t key_base,
    int64_t table_mask,
    int32_t date_lo,
    int32_t date_hi
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nli) {
        return;
    }

    int32_t ship_day = l_shipdate_days[idx];
    if (ship_day < date_lo || ship_day >= date_hi) {
        return;
    }

    uint64_t key = make_pair_key_i64(l_partkey[idx], l_suppkey[idx], key_base);
    uint64_t h = mix64(key) & (uint64_t)table_mask;

    while (true) {
        uint64_t found = table_keys[h];
        if (found == EMPTY_KEY) {
            return;
        }
        if (found == key) {
            int64_t ps_idx = table_vals[h];
            atomicAdd(&ps_sum_qty[ps_idx], l_quantity[idx]);
            return;
        }
        h = (h + 1ULL) & (uint64_t)table_mask;
    }
}

__global__ void mark_qualifying_supp_kernel(
    const int64_t* __restrict__ ps_suppkey,
    const int64_t* __restrict__ ps_availqty,
    const double* __restrict__ ps_sum_qty,
    uint8_t* __restrict__ supp_mask,
    int64_t nps,
    int64_t max_suppkey
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nps) {
        return;
    }

    double sum_qty = ps_sum_qty[idx];
    if (sum_qty > 0.0 && ((double)ps_availqty[idx]) > 0.5 * sum_qty) {
        int64_t supp = ps_suppkey[idx];
        if (supp >= 0 && supp <= max_suppkey) {
            supp_mask[supp] = (uint8_t)1;
        }
    }
}

__global__ void filter_supplier_kernel(
    const int64_t* __restrict__ s_suppkey,
    const uint8_t* __restrict__ supp_mask,
    int64_t* __restrict__ out_row_idx,
    int64_t* __restrict__ out_count,
    int64_t ns,
    int64_t max_suppkey
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ns) {
        return;
    }

    int64_t supp = s_suppkey[idx];
    if (supp >= 0 && supp <= max_suppkey && supp_mask[supp] != 0) {
        unsigned long long pos = atomicAdd((unsigned long long*)out_count, 1ULL);
        out_row_idx[(int64_t)pos] = idx;
    }
}

static int64_t next_power_of_two_i64(int64_t x) {
    int64_t p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

torch::Tensor q20_launcher(
    torch::Tensor ps_partkey,
    torch::Tensor ps_suppkey,
    torch::Tensor ps_availqty,
    torch::Tensor l_partkey,
    torch::Tensor l_suppkey,
    torch::Tensor l_quantity,
    torch::Tensor l_shipdate_days,
    torch::Tensor s_suppkey,
    int64_t max_suppkey,
    int32_t date_lo,
    int32_t date_hi
) {
    TORCH_CHECK(ps_partkey.scalar_type() == torch::kInt64, "ps_partkey must be int64");
    TORCH_CHECK(ps_suppkey.scalar_type() == torch::kInt64, "ps_suppkey must be int64");
    TORCH_CHECK(ps_availqty.scalar_type() == torch::kInt64, "ps_availqty must be int64");
    TORCH_CHECK(l_partkey.scalar_type() == torch::kInt64, "l_partkey must be int64");
    TORCH_CHECK(l_suppkey.scalar_type() == torch::kInt64, "l_suppkey must be int64");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32, "l_shipdate_days must be int32");
    TORCH_CHECK(s_suppkey.scalar_type() == torch::kInt64, "s_suppkey must be int64");

    if (!ps_partkey.is_cuda()) ps_partkey = ps_partkey.cuda();
    if (!ps_suppkey.is_cuda()) ps_suppkey = ps_suppkey.cuda();
    if (!ps_availqty.is_cuda()) ps_availqty = ps_availqty.cuda();
    if (!l_partkey.is_cuda()) l_partkey = l_partkey.cuda();
    if (!l_suppkey.is_cuda()) l_suppkey = l_suppkey.cuda();
    if (!l_quantity.is_cuda()) l_quantity = l_quantity.cuda();
    if (!l_shipdate_days.is_cuda()) l_shipdate_days = l_shipdate_days.cuda();
    if (!s_suppkey.is_cuda()) s_suppkey = s_suppkey.cuda();

    ps_partkey = ps_partkey.contiguous();
    ps_suppkey = ps_suppkey.contiguous();
    ps_availqty = ps_availqty.contiguous();
    l_partkey = l_partkey.contiguous();
    l_suppkey = l_suppkey.contiguous();
    l_quantity = l_quantity.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();
    s_suppkey = s_suppkey.contiguous();

    int64_t nps = ps_partkey.numel();
    int64_t nli = l_partkey.numel();
    int64_t ns = s_suppkey.numel();

    auto out_options = ps_partkey.options().dtype(torch::kInt64);
    if (nps == 0 || ns == 0 || max_suppkey < 0) {
        return torch::empty({0}, out_options);
    }

    TORCH_CHECK(ps_suppkey.numel() == nps && ps_availqty.numel() == nps, "partsupp tensors size mismatch");
    TORCH_CHECK(l_suppkey.numel() == nli && l_quantity.numel() == nli && l_shipdate_days.numel() == nli, "lineitem tensors size mismatch");

    int64_t hash_size = next_power_of_two_i64(nps * 4);
    if (hash_size < 1024) {
        hash_size = 1024;
    }
    int64_t table_mask = hash_size - 1;
    int64_t key_base = max_suppkey + 1;

    auto i64_options = ps_partkey.options().dtype(torch::kInt64);
    auto f64_options = l_quantity.options().dtype(torch::kFloat64);
    auto u8_options = ps_partkey.options().dtype(torch::kUInt8);

    torch::Tensor table_keys_t = torch::empty({hash_size}, i64_options);
    torch::Tensor table_vals_t = torch::empty({hash_size}, i64_options);
    torch::Tensor ps_sum_qty_t = torch::zeros({nps}, f64_options);
    torch::Tensor supp_mask_t = torch::zeros({max_suppkey + 1}, u8_options);
    torch::Tensor out_t = torch::empty({ns}, i64_options);
    torch::Tensor count_t = torch::zeros({1}, i64_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaMemsetAsync(table_keys_t.data_ptr<int64_t>(), 0xff, (size_t)hash_size * sizeof(int64_t), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_ps = (int)((nps + BLOCK_SIZE - 1) / BLOCK_SIZE);
    build_ps_hash_kernel<<<blocks_ps, BLOCK_SIZE, 0, stream>>>(
        ps_partkey.data_ptr<int64_t>(),
        ps_suppkey.data_ptr<int64_t>(),
        (uint64_t*)table_keys_t.data_ptr<int64_t>(),
        table_vals_t.data_ptr<int64_t>(),
        nps,
        key_base,
        table_mask
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (nli > 0) {
        int blocks_li = (int)((nli + BLOCK_SIZE - 1) / BLOCK_SIZE);
        scan_lineitem_accum_kernel<<<blocks_li, BLOCK_SIZE, 0, stream>>>(
            l_partkey.data_ptr<int64_t>(),
            l_suppkey.data_ptr<int64_t>(),
            l_quantity.data_ptr<double>(),
            l_shipdate_days.data_ptr<int32_t>(),
            (const uint64_t*)table_keys_t.data_ptr<int64_t>(),
            table_vals_t.data_ptr<int64_t>(),
            ps_sum_qty_t.data_ptr<double>(),
            nli,
            key_base,
            table_mask,
            date_lo,
            date_hi
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    mark_qualifying_supp_kernel<<<blocks_ps, BLOCK_SIZE, 0, stream>>>(
        ps_suppkey.data_ptr<int64_t>(),
        ps_availqty.data_ptr<int64_t>(),
        ps_sum_qty_t.data_ptr<double>(),
        supp_mask_t.data_ptr<uint8_t>(),
        nps,
        max_suppkey
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_s = (int)((ns + BLOCK_SIZE - 1) / BLOCK_SIZE);
    filter_supplier_kernel<<<blocks_s, BLOCK_SIZE, 0, stream>>>(
        s_suppkey.data_ptr<int64_t>(),
        supp_mask_t.data_ptr<uint8_t>(),
        out_t.data_ptr<int64_t>(),
        count_t.data_ptr<int64_t>(),
        ns,
        max_suppkey
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t h_count = 0;
    cudaMemcpyAsync(&h_count, count_t.data_ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out_t.slice(0, 0, h_count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q20_launcher", &q20_launcher, "TPC-H Q20 fused hash aggregate/filter CUDA launcher");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension from the embedded .cu source on disk.

    The device code performs the memory-bound Q20 lineitem scan and correlated
    partsupp aggregation in CUDA, avoiding PyTorch unique/scatter temporaries.
    """
    tmpdir = tempfile.mkdtemp(prefix="q20_cuda_")
    cu_path = os.path.join(tmpdir, "_q20_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q20_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-arch=sm_90", "--expt-relaxed-constexpr"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT: Any = _load_cuda()


def _empty_result(supplier_in_nation: cudf.DataFrame) -> cudf.DataFrame:
    """Return an empty cuDF result with the SQL projection columns."""
    return supplier_in_nation.head(0)[["s_name", "s_address"]].reset_index(drop=True)


def _series_to_torch_i64(series: cudf.Series) -> torch.Tensor:
    """Convert a cuDF numeric column to a CUDA int64 torch tensor without CPU materialization."""
    return torch.as_tensor(series.to_cupy(), dtype=torch.int64)


def _series_to_torch_f64(series: cudf.Series) -> torch.Tensor:
    """Convert a cuDF numeric column to a CUDA float64 torch tensor without CPU materialization."""
    return torch.as_tensor(series.to_cupy(), dtype=torch.float64)


def _series_to_torch_i32(series: cudf.Series) -> torch.Tensor:
    """Convert a cuDF numeric column to a CUDA int32 torch tensor without CPU materialization."""
    return torch.as_tensor(series.to_cupy(), dtype=torch.int32)


def _query_core(
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
    s_suppkey: torch.Tensor,
    max_suppkey: int,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """Run the Q20 correlated inventory test in CUDA.

    CUDA builds an open-addressed hash table for filtered partsupp pairs, scans
    lineitem once with coalesced date loads and conditional key/quantity loads,
    atomically accumulates quantity only for matching pairs, marks qualifying
    supplier keys, and returns row positions in the already nation-filtered
    supplier table. This replaces the reference PyTorch unique/scatter/is-in
    pipeline with one fused GPU hash-aggregate/filter path and fewer large
    intermediates.
    """
    return _EXT.q20_launcher(
        ps_partkey,
        ps_suppkey,
        ps_availqty,
        l_partkey,
        l_suppkey,
        l_quantity,
        l_shipdate_days,
        s_suppkey,
        int(max_suppkey),
        int(shipdate_lo_days),
        int(shipdate_hi_days),
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    name_prefix: str = "forest",
    nation_name: str = "CANADA",
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT s_name, s_address
        FROM supplier, nation
        WHERE s_suppkey IN (
            SELECT ps_suppkey
            FROM partsupp
            WHERE ps_partkey IN (
                SELECT p_partkey
                FROM part
                WHERE p_name LIKE 'forest%')
              AND ps_availqty > (
                SELECT 0.5 * sum(l_quantity)
                FROM lineitem
                WHERE l_partkey = ps_partkey
                  AND l_suppkey = ps_suppkey
                  AND l_shipdate >= DATE '1994-01-01'
                  AND l_shipdate <  DATE '1995-01-01'))
          AND s_nationkey = n_nationkey
          AND n_name = 'CANADA'
        ORDER BY s_name;

    Description:
        TPC-H Q20. cuDF handles string filtering, small dimension filtering, and
        the part-to-partsupp semijoin. The memory-bound correlated subquery is
        executed by a hand-written CUDA extension: it hashes candidate
        (ps_partkey, ps_suppkey) pairs, scans lineitem once for the date range,
        aggregates l_quantity into matching partsupp rows, marks qualifying
        suppliers, and filters the nation-qualified supplier rows on GPU. This
        avoids the reference PyTorch stack/cat/unique/scatter/is-in sequence and
        reduces large temporary tensors and global memory passes.

    Params:
        tables: mapping containing cuDF DataFrames for supplier, nation,
            partsupp, part, and lineitem.
        name_prefix: prefix filter for p_name, equivalent to LIKE '<prefix>%'.
        nation_name: nation filter for suppliers.
        shipdate_lo: inclusive lower ship date bound.
        shipdate_hi: exclusive upper ship date bound.
    """
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_partsupp = tables["partsupp"]
    df_part = tables["part"]
    df_lineitem = tables["lineitem"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    part_mask = df_part["p_name"].str.startswith(name_prefix)
    filtered_part = df_part[part_mask][["p_partkey"]]

    filtered_partsupp = df_partsupp.merge(
        filtered_part,
        left_on="ps_partkey",
        right_on="p_partkey",
        how="inner",
    )[["ps_partkey", "ps_suppkey", "ps_availqty"]]

    nation_filtered = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supplier_in_nation = df_supplier.merge(
        nation_filtered,
        left_on="s_nationkey",
        right_on="n_nationkey",
        how="inner",
    ).reset_index(drop=True)

    if len(filtered_partsupp) == 0 or len(supplier_in_nation) == 0 or len(df_lineitem) == 0:
        return _empty_result(supplier_in_nation)

    max_suppkey = int(df_supplier["s_suppkey"].max())
    if max_suppkey < 0:
        return _empty_result(supplier_in_nation)

    l_shipdate_days = (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")

    l_partkey_t = _series_to_torch_i64(df_lineitem["l_partkey"])
    l_suppkey_t = _series_to_torch_i64(df_lineitem["l_suppkey"])
    l_quantity_t = _series_to_torch_f64(df_lineitem["l_quantity"])
    l_shipdate_days_t = _series_to_torch_i32(l_shipdate_days)

    ps_partkey_t = _series_to_torch_i64(filtered_partsupp["ps_partkey"])
    ps_suppkey_t = _series_to_torch_i64(filtered_partsupp["ps_suppkey"])
    ps_availqty_t = _series_to_torch_i64(filtered_partsupp["ps_availqty"])

    s_suppkey_t = _series_to_torch_i64(supplier_in_nation["s_suppkey"])

    selected_idx_t = _query_core(
        l_partkey_t,
        l_suppkey_t,
        l_quantity_t,
        l_shipdate_days_t,
        ps_partkey_t,
        ps_suppkey_t,
        ps_availqty_t,
        s_suppkey_t,
        max_suppkey,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    if selected_idx_t.numel() == 0:
        return _empty_result(supplier_in_nation)

    selected_idx_cudf = cudf.Series(selected_idx_t)
    try:
        result = supplier_in_nation.take(selected_idx_cudf)[["s_name", "s_address"]]
    except Exception:
        result = supplier_in_nation.iloc[selected_idx_cudf.to_pandas().tolist()][["s_name", "s_address"]]

    result = result.sort_values("s_name").reset_index(drop=True)
    return result
