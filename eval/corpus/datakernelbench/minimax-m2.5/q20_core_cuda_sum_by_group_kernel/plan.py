import os
import tempfile

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 1024

//----------------------------------------------------------------------
// Kernel: sum_quantity_by_group
// Each thread reads one lineitem row (group_id, quantity) and atomically
// adds the quantity to the corresponding group sum in global memory.
//----------------------------------------------------------------------
__global__ void sum_by_group_kernel(
    const int64_t* __restrict__ group_ids,
    const double* __restrict__ quantities,
    double* __restrict__ output,
    int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t grp = group_ids[idx];
        double val = quantities[idx];
        atomicAdd(&output[grp], val);
    }
}

//----------------------------------------------------------------------
// Host launcher for sum_by_group
//----------------------------------------------------------------------
void sum_by_group(torch::Tensor group_ids,
                  torch::Tensor quantities,
                  torch::Tensor output) {
    TORCH_CHECK(group_ids.is_contiguous(), "group_ids must be contiguous");
    TORCH_CHECK(quantities.is_contiguous(), "quantities must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(group_ids.is_cuda(), "group_ids must be a CUDA tensor");
    TORCH_CHECK(quantities.is_cuda(), "quantities must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");

    int64_t N = group_ids.numel();
    const int block_size = BLOCK_SIZE;
    const int grid_size = (N + block_size - 1) / block_size;

    sum_by_group_kernel<<<grid_size, block_size>>>(
        group_ids.data_ptr<int64_t>(),
        quantities.data_ptr<double>(),
        output.data_ptr<double>(),
        N
    );
    cudaDeviceSynchronize();
}

//----------------------------------------------------------------------
// Kernel: filter_and_collect_suppkeys
// For each partsupp row, check the condition:
//   ps_availqty > 0.5 * sum_qty[group]  AND  sum_qty[group] > 0
// If true, write the suppkey to output; otherwise write -1.
//----------------------------------------------------------------------
__global__ void filter_and_collect_suppkeys_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ suppkeys,
    const int64_t* __restrict__ availqty,
    const double* __restrict__ sum_qty,
    int64_t* __restrict__ output,
    int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t grp = group_ids[idx];
        double sum = sum_qty[grp];
        if (sum > 0.0) {
            double half_sum = 0.5 * sum;
            double avail = static_cast<double>(availqty[idx]);
            if (avail > half_sum) {
                output[idx] = suppkeys[idx];
                return;
            }
        }
        output[idx] = -1;
    }
}

//----------------------------------------------------------------------
// Host launcher for filter_and_collect_suppkeys
//----------------------------------------------------------------------
void filter_and_collect_suppkeys(torch::Tensor group_ids,
                                 torch::Tensor suppkeys,
                                 torch::Tensor availqty,
                                 torch::Tensor sum_qty,
                                 torch::Tensor output) {
    TORCH_CHECK(group_ids.is_contiguous(), "group_ids must be contiguous");
    TORCH_CHECK(suppkeys.is_contiguous(), "suppkeys must be contiguous");
    TORCH_CHECK(availqty.is_contiguous(), "availqty must be contiguous");
    TORCH_CHECK(sum_qty.is_contiguous(), "sum_qty must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(group_ids.is_cuda(), "group_ids must be CUDA");
    TORCH_CHECK(suppkeys.is_cuda(), "suppkeys must be CUDA");
    TORCH_CHECK(availqty.is_cuda(), "availqty must be CUDA");
    TORCH_CHECK(sum_qty.is_cuda(), "sum_qty must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");

    int64_t N = group_ids.numel();
    const int block_size = BLOCK_SIZE;
    const int grid_size = (N + block_size - 1) / block_size;

    filter_and_collect_suppkeys_kernel<<<grid_size, block_size>>>(
        group_ids.data_ptr<int64_t>(),
        suppkeys.data_ptr<int64_t>(),
        availqty.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        output.data_ptr<int64_t>(),
        N
    );
    cudaDeviceSynchronize();
}

//----------------------------------------------------------------------
// PyBind11 bindings
//----------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_by_group", &sum_by_group,
          "Sum lineitem quantities by (partkey,suppkey) group (CUDA)");
    m.def("filter_and_collect_suppkeys", &filter_and_collect_suppkeys,
          "Filter partsupp rows based on inventory condition (CUDA)");
}
"""

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q20_cuda_")
    cu_path = os.path.join(tmpdir, "_q20_kernels.cu")
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
    )

_EXT = _load_cuda()


def _ensure_cuda(t: torch.Tensor) -> torch.Tensor:
    """Move tensor to CUDA and make contiguous if needed."""
    if t.is_cuda:
        return t.contiguous()
    return t.contiguous().cuda()


def _sum_by_group_cuda(group_ids: torch.Tensor, quantities: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Compute sum(quantities) per group using a custom CUDA kernel."""
    group_ids = _ensure_cuda(group_ids)
    quantities = _ensure_cuda(quantities)
    output = torch.zeros(num_groups, dtype=torch.float64, device='cuda')
    _EXT.sum_by_group(group_ids, quantities, output)
    return output


def _filter_suppkeys_cuda(ps_group: torch.Tensor,
                          ps_suppkey: torch.Tensor,
                          ps_availqty: torch.Tensor,
                          sum_qty: torch.Tensor) -> torch.Tensor:
    """Filter partsupp rows based on inventory condition, returning suppkeys or -1."""
    ps_group = _ensure_cuda(ps_group)
    ps_suppkey = _ensure_cuda(ps_suppkey)
    ps_availqty = _ensure_cuda(ps_availqty)
    sum_qty = _ensure_cuda(sum_qty)
    N = ps_group.shape[0]
    out = torch.full((N,), -1, dtype=torch.int64, device='cuda')
    _EXT.filter_and_collect_suppkeys(ps_group, ps_suppkey, ps_availqty, sum_qty, out)
    return out


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
    GPU‑accelerated implementation of the hot path for TPC‑H Q20.
    Performs:
      1. Build composite (partkey, suppkey) pairs for lineitem and partsupp.
      2. Compute unique pairs and obtain group IDs via torch.unique.
      3. Sum l_quantity per group using a custom CUDA kernel (atomic add).
      4. Evaluate the inventory condition for each partsupp row using a custom CUDA kernel.
      5. Collect qualifying suppkeys, compute unique set, and mask supplier rows.

    This replaces the original PyTorch scatter‑add / unique / isin sequence with
    two fused CUDA kernels, reducing kernel launch overhead and improving memory
    bandwidth utilization.
    """
    # Ensure all input tensors are on the GPU for the subsequent ops
    l_partkey = _ensure_cuda(l_partkey)
    l_suppkey = _ensure_cuda(l_suppkey)
    l_quantity = _ensure_cuda(l_quantity)
    ps_partkey = _ensure_cuda(ps_partkey)
    ps_suppkey = _ensure_cuda(ps_suppkey)
    ps_availqty = _ensure_cuda(ps_availqty)
    s_suppkey = _ensure_cuda(s_suppkey)
    s_row_idx = _ensure_cuda(s_row_idx)

    # -----------------------------------------------------------------
    # 1) Build composite keys and compute group IDs via torch.unique
    # -----------------------------------------------------------------
    ps_pair = torch.stack([ps_partkey, ps_suppkey], dim=1)       # [N_ps, 2]
    li_pair = torch.stack([l_partkey, l_suppkey], dim=1)         # [N_li, 2]

    all_pairs = torch.cat([ps_pair, li_pair], dim=0)             # [N_ps + N_li, 2]

    # torch.unique returns sorted unique keys and an inverse index mapping
    unique_pairs, inverse = torch.unique(all_pairs, dim=0, return_inverse=True)
    Nps = ps_pair.shape[0]
    ps_group = inverse[:Nps]        # group IDs for partsupp rows
    li_group = inverse[Nps:]        # group IDs for lineitem rows

    num_groups = unique_pairs.shape[0]

    # -----------------------------------------------------------------
    # 2) Sum l_quantity per group using custom CUDA kernel
    # -----------------------------------------------------------------
    sum_qty = _sum_by_group_cuda(li_group, l_quantity, num_groups)

    # -----------------------------------------------------------------
    # 3) Evaluate inventory condition for each partsupp row
    # -----------------------------------------------------------------
    qual_suppkeys_tensor = _filter_suppkeys_cuda(ps_group, ps_suppkey, ps_availqty, sum_qty)

    # -----------------------------------------------------------------
    # 4) Collect qualifying suppkeys and compute unique set
    # -----------------------------------------------------------------
    qual_mask = qual_suppkeys_tensor != -1
    qual_suppkeys = qual_suppkeys_tensor[qual_mask]
    unique_qual_supp = torch.unique(qual_suppkeys)

    # -----------------------------------------------------------------
    # 5) Mask supplier rows and return row indices
    # -----------------------------------------------------------------
    supp_mask = torch.isin(s_suppkey, unique_qual_supp)
    return s_row_idx[supp_mask]


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
