"""TPC-H Q20 core optimized with a hand-written CUDA extension.

run_query remains the reference cuDF/PyTorch data-preparation path: cuDF performs
string/date filters and joins, then tensors are passed to _query_core.  The hot
numeric core runs on the GPU in CUDA: it builds a hash table for filtered
partsupp keys, scans filtered lineitem once to aggregate matching quantities,
builds a qualifying supplier-key set, and selects supplier row indices.  This is
faster than the reference torch.unique/scatter/is-in pipeline because it avoids
large composite-key materialization and global sort/unique work, uses coalesced
1D scans, conditionally loads l_quantity only for matching keys, and keeps all
membership/aggregation work in fused CUDA kernels with fewer temporaries.
"""

import os
import tempfile

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

#define BLOCK_SIZE 256

__device__ __forceinline__ uint64_t mix_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t hash_pair_i64(int64_t a, int64_t b) {
    uint64_t x = mix_u64((uint64_t)a);
    uint64_t y = mix_u64((uint64_t)b + 0x9e3779b97f4a7c15ULL);
    return x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2));
}

__device__ __forceinline__ uint64_t hash_i64(int64_t a) {
    return mix_u64((uint64_t)a);
}

__global__ void insert_partsupp_kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    int64_t* __restrict__ ht_part,
    int64_t* __restrict__ ht_supp,
    int64_t* __restrict__ ht_idx,
    int64_t nps,
    int64_t cap
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nps) return;

    int64_t p = ps_partkey[i];
    int64_t s = ps_suppkey[i];
    uint64_t mask = (uint64_t)(cap - 1);
    uint64_t pos = hash_pair_i64(p, s) & mask;

    for (int64_t probe = 0; probe < cap; ++probe) {
        int64_t old = (int64_t)atomicCAS(
            (unsigned long long*)&ht_idx[pos],
            0xffffffffffffffffULL,
            (unsigned long long)i
        );
        if (old == -1) {
            ht_part[pos] = p;
            ht_supp[pos] = s;
            return;
        }
        pos = (pos + 1) & mask;
    }
}

__global__ void aggregate_lineitem_kernel(
    const int64_t* __restrict__ l_partkey,
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_quantity,
    const int64_t* __restrict__ ht_part,
    const int64_t* __restrict__ ht_supp,
    const int64_t* __restrict__ ht_idx,
    double* __restrict__ sums,
    int64_t nli,
    int64_t cap
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nli) return;

    int64_t p = l_partkey[i];
    int64_t s = l_suppkey[i];
    uint64_t mask = (uint64_t)(cap - 1);
    uint64_t pos = hash_pair_i64(p, s) & mask;

    for (int64_t probe = 0; probe < cap; ++probe) {
        int64_t slot_idx = ht_idx[pos];
        if (slot_idx == -1) {
            return;
        }
        if (ht_part[pos] == p && ht_supp[pos] == s) {
            double q = l_quantity[i];
            atomicAdd(&sums[slot_idx], q);
            return;
        }
        pos = (pos + 1) & mask;
    }
}

__global__ void qualify_partsupp_kernel(
    const int64_t* __restrict__ ps_suppkey,
    const int64_t* __restrict__ ps_availqty,
    const double* __restrict__ sums,
    int64_t* __restrict__ qual_supp_set,
    int64_t nps,
    int64_t supp_cap
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nps) return;

    double sum_q = sums[i];
    if (!(sum_q > 0.0)) return;

    double avail = (double)ps_availqty[i];
    if (!(avail > 0.5 * sum_q)) return;

    int64_t s = ps_suppkey[i];
    uint64_t mask = (uint64_t)(supp_cap - 1);
    uint64_t pos = hash_i64(s) & mask;

    for (int64_t probe = 0; probe < supp_cap; ++probe) {
        int64_t old = (int64_t)atomicCAS(
            (unsigned long long*)&qual_supp_set[pos],
            0xffffffffffffffffULL,
            (unsigned long long)s
        );
        if (old == -1 || old == s) {
            return;
        }
        pos = (pos + 1) & mask;
    }
}

__global__ void select_supplier_kernel(
    const int64_t* __restrict__ s_suppkey,
    const int64_t* __restrict__ s_row_idx,
    const int64_t* __restrict__ qual_supp_set,
    int64_t* __restrict__ out_idx,
    int64_t* __restrict__ out_count,
    int64_t ns,
    int64_t supp_cap
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns) return;

    int64_t s = s_suppkey[i];
    uint64_t mask = (uint64_t)(supp_cap - 1);
    uint64_t pos = hash_i64(s) & mask;

    for (int64_t probe = 0; probe < supp_cap; ++probe) {
        int64_t v = qual_supp_set[pos];
        if (v == -1) {
            return;
        }
        if (v == s) {
            unsigned long long write_pos = atomicAdd(
                (unsigned long long*)out_count,
                1ULL
            );
            out_idx[(int64_t)write_pos] = s_row_idx[i];
            return;
        }
        pos = (pos + 1) & mask;
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
    torch::Tensor l_partkey,
    torch::Tensor l_suppkey,
    torch::Tensor l_quantity,
    torch::Tensor ps_partkey,
    torch::Tensor ps_suppkey,
    torch::Tensor ps_availqty,
    torch::Tensor s_suppkey,
    torch::Tensor s_row_idx
) {
    TORCH_CHECK(l_partkey.is_cuda(), "l_partkey must be CUDA");
    TORCH_CHECK(l_suppkey.is_cuda(), "l_suppkey must be CUDA");
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be CUDA");
    TORCH_CHECK(ps_partkey.is_cuda(), "ps_partkey must be CUDA");
    TORCH_CHECK(ps_suppkey.is_cuda(), "ps_suppkey must be CUDA");
    TORCH_CHECK(ps_availqty.is_cuda(), "ps_availqty must be CUDA");
    TORCH_CHECK(s_suppkey.is_cuda(), "s_suppkey must be CUDA");
    TORCH_CHECK(s_row_idx.is_cuda(), "s_row_idx must be CUDA");

    TORCH_CHECK(l_partkey.scalar_type() == torch::kInt64, "l_partkey must be int64");
    TORCH_CHECK(l_suppkey.scalar_type() == torch::kInt64, "l_suppkey must be int64");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(ps_partkey.scalar_type() == torch::kInt64, "ps_partkey must be int64");
    TORCH_CHECK(ps_suppkey.scalar_type() == torch::kInt64, "ps_suppkey must be int64");
    TORCH_CHECK(ps_availqty.scalar_type() == torch::kInt64, "ps_availqty must be int64");
    TORCH_CHECK(s_suppkey.scalar_type() == torch::kInt64, "s_suppkey must be int64");
    TORCH_CHECK(s_row_idx.scalar_type() == torch::kInt64, "s_row_idx must be int64");

    l_partkey = l_partkey.contiguous();
    l_suppkey = l_suppkey.contiguous();
    l_quantity = l_quantity.contiguous();
    ps_partkey = ps_partkey.contiguous();
    ps_suppkey = ps_suppkey.contiguous();
    ps_availqty = ps_availqty.contiguous();
    s_suppkey = s_suppkey.contiguous();
    s_row_idx = s_row_idx.contiguous();

    int64_t nli = l_partkey.numel();
    int64_t nps = ps_partkey.numel();
    int64_t ns = s_suppkey.numel();

    TORCH_CHECK(l_suppkey.numel() == nli && l_quantity.numel() == nli, "lineitem tensor length mismatch");
    TORCH_CHECK(ps_suppkey.numel() == nps && ps_availqty.numel() == nps, "partsupp tensor length mismatch");
    TORCH_CHECK(s_row_idx.numel() == ns, "supplier tensor length mismatch");

    if (nps == 0 || ns == 0 || nli == 0) {
        return torch::empty({0}, s_row_idx.options());
    }

    int64_t pair_cap = next_power_of_two_i64(std::max<int64_t>(16, nps * 4));
    int64_t supp_cap = pair_cap;

    auto int_opts = ps_partkey.options();
    auto dbl_opts = l_quantity.options();

    auto ht_part = torch::empty({pair_cap}, int_opts);
    auto ht_supp = torch::empty({pair_cap}, int_opts);
    auto ht_idx = torch::full({pair_cap}, -1, int_opts);
    auto sums = torch::zeros({nps}, dbl_opts);
    auto qual_supp_set = torch::full({supp_cap}, -1, int_opts);
    auto out_idx = torch::empty({ns}, s_row_idx.options());
    auto out_count = torch::zeros({1}, s_row_idx.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int blocks_ps = (int)((nps + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int blocks_li = (int)((nli + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int blocks_s = (int)((ns + BLOCK_SIZE - 1) / BLOCK_SIZE);

    insert_partsupp_kernel<<<blocks_ps, BLOCK_SIZE, 0, stream>>>(
        ps_partkey.data_ptr<int64_t>(),
        ps_suppkey.data_ptr<int64_t>(),
        ht_part.data_ptr<int64_t>(),
        ht_supp.data_ptr<int64_t>(),
        ht_idx.data_ptr<int64_t>(),
        nps,
        pair_cap
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    aggregate_lineitem_kernel<<<blocks_li, BLOCK_SIZE, 0, stream>>>(
        l_partkey.data_ptr<int64_t>(),
        l_suppkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        ht_part.data_ptr<int64_t>(),
        ht_supp.data_ptr<int64_t>(),
        ht_idx.data_ptr<int64_t>(),
        sums.data_ptr<double>(),
        nli,
        pair_cap
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    qualify_partsupp_kernel<<<blocks_ps, BLOCK_SIZE, 0, stream>>>(
        ps_suppkey.data_ptr<int64_t>(),
        ps_availqty.data_ptr<int64_t>(),
        sums.data_ptr<double>(),
        qual_supp_set.data_ptr<int64_t>(),
        nps,
        supp_cap
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    select_supplier_kernel<<<blocks_s, BLOCK_SIZE, 0, stream>>>(
        s_suppkey.data_ptr<int64_t>(),
        s_row_idx.data_ptr<int64_t>(),
        qual_supp_set.data_ptr<int64_t>(),
        out_idx.data_ptr<int64_t>(),
        out_count.data_ptr<int64_t>(),
        ns,
        supp_cap
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t h_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &h_count,
        out_count.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return out_idx.narrow(0, 0, h_count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q20_launcher", &q20_launcher, "TPC-H Q20 tensor core using CUDA hash aggregation");
}
'''


def _load_cuda() -> object:
    """Compile and load the CUDA extension from an embedded .cu source string.

    Host Python writes the device code to a temporary on-disk file because
    torch.utils.cpp_extension.load requires a real source path.  All query hot
    path work is then executed by GPU CUDA kernels, not by Python loops.
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
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )


_EXT = _load_cuda()


def _query_core(
    # lineitem filtered tensors (already filtered by shipdate range in run_query)
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    # partsupp tensors (already filtered by part name in run_query)
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
    # supplier (already filtered by nation = CANADA in run_query)
    s_suppkey: torch.Tensor,
    s_row_idx: torch.Tensor,
) -> torch.Tensor:
    """Run TPC-H Q20's numeric semi-join/aggregation core on the GPU.

    CUDA builds an open-addressed hash table for filtered (ps_partkey,
    ps_suppkey), scans date-filtered lineitem once to accumulate matching
    l_quantity in float64, inserts qualifying ps_suppkey values into a GPU hash
    set, and emits matching supplier row indices.  This replaces the reference
    torch.unique/scatter/is-in sequence with hash aggregation and membership
    kernels, reducing temporary tensors and avoiding sort-like unique work.
    """
    return _EXT.q20_launcher(
        l_partkey,
        l_suppkey,
        l_quantity,
        ps_partkey,
        ps_suppkey,
        ps_availqty,
        s_suppkey,
        s_row_idx,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    name_prefix: str = "forest",
    nation_name: str = "CANADA",
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT s_name, s_address FROM supplier, nation WHERE s_suppkey IN (
            SELECT ps_suppkey FROM partsupp WHERE ps_partkey IN (
                SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
            AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem
                               WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
                               AND l_shipdate >= DATE '1994-01-01'
                               AND l_shipdate <  DATE '1995-01-01'))
          AND s_nationkey = n_nationkey AND n_name = 'CANADA'
          ORDER BY s_name;
    Description: TPC-H Q20 — finds suppliers in a specific nation who have excess inventory
                 of parts whose name starts with a given prefix during a date range.
    Params:
        name_prefix (str): prefix filter for p_name (LIKE '<prefix>%').
        nation_name (str): nation filter for supplier's nation.
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_partsupp = tables["partsupp"]
    df_part = tables["part"]
    df_lineitem = tables["lineitem"]

    # Convert date parameters to epoch days (int32)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # --- Filter part by p_name LIKE 'prefix%' ---
    part_mask = df_part["p_name"].str.startswith(name_prefix)
    filtered_part = df_part[part_mask][["p_partkey"]]

    # --- Filter partsupp by ps_partkey IN filtered_part ---
    filtered_partsupp = df_partsupp.merge(
        filtered_part, left_on="ps_partkey", right_on="p_partkey", how="inner"
    )[["ps_partkey", "ps_suppkey", "ps_availqty"]]

    # --- Filter lineitem by shipdate range ---
    l_shipdate_days = (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")
    li_mask = (l_shipdate_days >= shipdate_lo_days) & (l_shipdate_days < shipdate_hi_days)
    filtered_lineitem = df_lineitem[li_mask][["l_partkey", "l_suppkey", "l_quantity"]]

    # --- Filter supplier by nation = CANADA ---
    nation_filtered = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supplier_in_nation = df_supplier.merge(
        nation_filtered, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    ).reset_index(drop=True)

    # Add a row index column so we can recover s_name / s_address by tensor index
    supplier_in_nation = supplier_in_nation.reset_index(drop=True)
    s_row_idx_series = cudf.Series(
        cudf.core.column.arange(0, len(supplier_in_nation), dtype="int64")
    ) if False else cudf.Series(range(len(supplier_in_nation))).astype("int64")

    # Convert tensors
    l_partkey_t = torch.as_tensor(filtered_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_suppkey_t = torch.as_tensor(filtered_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(filtered_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    ps_partkey_t = torch.as_tensor(filtered_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(filtered_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    ps_availqty_t = torch.as_tensor(filtered_partsupp["ps_availqty"].to_cupy(), dtype=torch.int64)

    s_suppkey_t = torch.as_tensor(supplier_in_nation["s_suppkey"].to_cupy(), dtype=torch.int64)
    s_row_idx_t = torch.as_tensor(s_row_idx_series.to_cupy(), dtype=torch.int64)

    selected_idx_t = _query_core(
        l_partkey_t, l_suppkey_t, l_quantity_t,
        ps_partkey_t, ps_suppkey_t, ps_availqty_t,
        s_suppkey_t, s_row_idx_t,
    )

    # Materialize result via gather on cuDF side
    selected_idx_cudf = cudf.Series(selected_idx_t)
    result = supplier_in_nation.iloc[selected_idx_cudf.to_pandas().tolist()][["s_name", "s_address"]]
    result = result.sort_values("s_name").reset_index(drop=True)
    return result
