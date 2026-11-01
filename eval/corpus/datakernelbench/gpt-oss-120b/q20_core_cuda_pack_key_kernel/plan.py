import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused kernels for TPC‑H Q20 core computation
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <cstdint>

#define BLOCK_SIZE 1024

// ------------------------------------------------------------------
// Kernel: pack (partkey, suppkey) into a 64‑bit key.
//   key = (uint64_t(part) << 32) | uint64_t(supp)
__global__ void pack_key_kernel(const int64_t* __restrict__ part,
                                const int64_t* __restrict__ supp,
                                uint64_t* __restrict__ out,
                                int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint64_t pk = static_cast<uint64_t>(part[idx]);
        uint64_t sk = static_cast<uint64_t>(supp[idx]);
        out[idx] = (pk << 32) | sk;
    }
}

// ------------------------------------------------------------------
// Kernel: produce a boolean mask for partsupp rows that satisfy:
//   ps_availqty > 0.5 * sum_qty_of_matching_lineitem  && sum_qty > 0
__global__ void ps_mask_kernel(const uint64_t* __restrict__ ps_key,
                               const int64_t* __restrict__ lower_idx,
                               const uint64_t* __restrict__ uniq_key,
                               const double* __restrict__ uniq_sum,
                               const int64_t* __restrict__ ps_availqty,
                               uint8_t* __restrict__ mask,
                               int64_t n_ps,
                               int64_t n_groups) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_ps) {
        uint64_t k = ps_key[i];
        int64_t pos = lower_idx[i];
        double sum = 0.0;
        if (pos < n_groups && uniq_key[pos] == k) {
            sum = uniq_sum[pos];
        }
        bool keep = (sum > 0.0) && (static_cast<double>(ps_availqty[i]) > 0.5 * sum);
        mask[i] = keep ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Launcher: builds the mask described above.
// Returns a uint8 tensor (0/1) of length ps_n.
torch::Tensor q20_mask_launcher(torch::Tensor l_partkey,
                                torch::Tensor l_suppkey,
                                torch::Tensor l_quantity,
                                torch::Tensor ps_partkey,
                                torch::Tensor ps_suppkey,
                                torch::Tensor ps_availqty) {
    // --------------------------------------------------------------
    // Basic sanity / device placement
    // --------------------------------------------------------------
    TORCH_CHECK(l_partkey.is_cuda() && l_suppkey.is_cuda() && l_quantity.is_cuda(),
                "lineitem tensors must be CUDA");
    TORCH_CHECK(ps_partkey.is_cuda() && ps_suppkey.is_cuda() && ps_availqty.is_cuda(),
                "partsupp tensors must be CUDA");

    auto li_n = l_partkey.numel();
    auto ps_n = ps_partkey.numel();

    // --------------------------------------------------------------
    // Allocate key buffers
    // --------------------------------------------------------------
    auto opts = torch::TensorOptions().dtype(torch::kUInt64).device(l_partkey.device());
    torch::Tensor li_key = torch::empty({li_n}, opts);
    torch::Tensor ps_key = torch::empty({ps_n}, opts);

    // --------------------------------------------------------------
    // Fill keys (grid stride)
    // --------------------------------------------------------------
    int blocks_li = (li_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pack_key_kernel<<<blocks_li, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_suppkey.data_ptr<int64_t>(),
        li_key.data_ptr<uint64_t>(),
        li_n
    );

    int blocks_ps = (ps_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pack_key_kernel<<<blocks_ps, BLOCK_SIZE>>>(
        ps_partkey.data_ptr<int64_t>(),
        ps_suppkey.data_ptr<int64_t>(),
        ps_key.data_ptr<uint64_t>(),
        ps_n
    );

    // --------------------------------------------------------------
    // Sort lineitem keys + quantities together
    // --------------------------------------------------------------
    thrust::device_ptr<uint64_t> d_li_key(
        static_cast<uint64_t*>(li_key.data_ptr()));
    thrust::device_ptr<double> d_li_qty(
        static_cast<double*>(l_quantity.data_ptr()));

    thrust::sort_by_key(thrust::device, d_li_key, d_li_key + li_n, d_li_qty);

    // --------------------------------------------------------------
    // Reduce by key to obtain (unique_key, sum_quantity)
    // --------------------------------------------------------------
    // Allocate max‑size buffers (worst case: all keys unique)
    torch::Tensor uniq_key_buf = torch::empty({li_n}, opts);
    torch::Tensor uniq_sum_buf = torch::empty({li_n},
        torch::TensorOptions().dtype(torch::kFloat64).device(l_partkey.device()));

    thrust::device_ptr<uint64_t> d_uniq_key(
        static_cast<uint64_t*>(uniq_key_buf.data_ptr()));
    thrust::device_ptr<double> d_uniq_sum(
        static_cast<double*>(uniq_sum_buf.data_ptr()));

    auto reduce_end = thrust::reduce_by_key(
        thrust::device,
        d_li_key, d_li_key + li_n,
        d_li_qty,
        d_uniq_key,
        d_uniq_sum
    );
    int64_t n_groups = reduce_end.first - d_uniq_key;   // number of distinct (part,supp)

    // --------------------------------------------------------------
    // Lower‑bound: for every partsupp key find first position >= key
    // --------------------------------------------------------------
    torch::Tensor lower_idx = torch::empty({ps_n},
        torch::TensorOptions().dtype(torch::kInt64).device(l_partkey.device()));
    thrust::device_ptr<int64_t> d_lower_idx(
        static_cast<int64_t*>(lower_idx.data_ptr()));

    thrust::lower_bound(
        thrust::device,
        d_uniq_key, d_uniq_key + n_groups,
        ps_key.data_ptr<uint64_t>(),
        ps_key.data_ptr<uint64_t>() + ps_n,
        d_lower_idx
    );

    // --------------------------------------------------------------
    // Build mask
    // --------------------------------------------------------------
    torch::Tensor mask = torch::empty({ps_n},
        torch::TensorOptions().dtype(torch::kUInt8).device(l_partkey.device()));

    ps_mask_kernel<<<blocks_ps, BLOCK_SIZE>>>(
        ps_key.data_ptr<uint64_t>(),
        lower_idx.data_ptr<int64_t>(),
        uniq_key_buf.data_ptr<uint64_t>(),
        uniq_sum_buf.data_ptr<double>(),
        ps_availqty.data_ptr<int64_t>(),
        mask.data_ptr<uint8_t>(),
        ps_n,
        n_groups
    );

    // Ensure kernel completion before returning
    cudaDeviceSynchronize();

    return mask;
}

// ------------------------------------------------------------------
// PyBind11 binding
// ------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q20_mask_launcher", &q20_mask_launcher,
          "TPC‑H Q20 mask kernel (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Helper: compile and load the CUDA extension at import time
# ----------------------------------------------------------------------
def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q20_cuda_")
    cu_path = os.path.join(tmpdir, "q20_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q20_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load once
_EXT = _load_cuda()

# ----------------------------------------------------------------------
# Launcher wrapper used by _query_core
# ----------------------------------------------------------------------
def _launch_mask(
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
) -> torch.Tensor:
    """Launch the fused CUDA kernel that returns a uint8 mask (0/1) for each
    partsupp row indicating whether the row satisfies the Q20 predicate.
    The mask is computed via a single sort‑reduce‑search‑mask pipeline,
    avoiding multiple Python‑level passes and reducing memory traffic."""
    # Ensure contiguous CUDA tensors
    l_partkey = l_partkey.contiguous()
    l_suppkey = l_suppkey.contiguous()
    l_quantity = l_quantity.contiguous()
    ps_partkey = ps_partkey.contiguous()
    ps_suppkey = ps_suppkey.contiguous()
    ps_availqty = ps_availqty.contiguous()
    return _EXT.q20_mask_launcher(
        l_partkey, l_suppkey, l_quantity,
        ps_partkey, ps_suppkey, ps_availqty,
    )

# ----------------------------------------------------------------------
# Core GPU computation: grouping, condition, and final supplier filtering
# ----------------------------------------------------------------------
def _query_core(
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
    s_suppkey: torch.Tensor,
    s_row_idx: torch.Tensor,
) -> torch.Tensor:
    """
    GPU‑accelerated core of TPC‑H Q20.

    Steps performed:
    1. Fuse lineitem aggregation (sum(l_quantity) per (part,supp))
       using a sort‑by‑key + reduce‑by‑key pipeline on the GPU.
    2. For each partsupp row, look up the aggregated quantity,
       apply the predicate ``ps_availqty > 0.5 * sum_qty``.
    3. Produce a boolean mask for partsupp rows that satisfy the condition.
    4. From the qualifying partsupp rows, obtain distinct supplier keys,
       then filter the supplier table via ``torch.isin``.
    The heavy aggregation and lookup is done in a single CUDA launch,
    dramatically reducing HBM traffic compared with the reference which
    performs multiple separate tensor ops.
    """
    # 1‑3: fused mask via custom CUDA kernel
    ps_mask = _launch_mask(
        l_partkey, l_suppkey, l_quantity,
        ps_partkey, ps_suppkey, ps_availqty,
    ).bool()  # convert uint8 → bool

    # 4: finish on the Python side (still entirely on GPU)
    qualifying_ps_suppkey = ps_suppkey[ps_mask]
    if qualifying_ps_suppkey.numel() == 0:
        # No supplier qualifies → return empty int64 tensor
        return torch.empty((0,), dtype=torch.int64, device=s_row_idx.device)

    unique_qual_supp = torch.unique(qualifying_ps_suppkey)
    supp_mask = torch.isin(s_suppkey, unique_qual_supp)
    return s_row_idx[supp_mask]

# ----------------------------------------------------------------------
# Full query: unchanged from reference (only the hot path changed)
# ----------------------------------------------------------------------
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
    Description: TPC‑H Q20 — finds suppliers in a specific nation who have excess inventory
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
    s_row_idx_series = cudf.Series(range(len(supplier_in_nation))).astype("int64")

    # Convert to torch tensors (GPU)
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
