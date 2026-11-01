import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q21_kernel(
    const int64_t* __restrict__ l1_orderkey_ptr,
    const int64_t* __restrict__ l1_suppkey_ptr,
    const bool* __restrict__ l1_late_ptr,
    const int64_t* __restrict__ supp_ids_ptr,
    const int64_t* __restrict__ lo_orderkeys_ptr,
    const int64_t* __restrict__ lo_distinct_supp_ptr,
    const int64_t* __restrict__ lo_late_distinct_supp_ptr,
    int64_t num_suppliers,
    int* __restrict__ out_counts_ptr,
    int64_t n_elements,
    int64_t n_lo
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        int64_t l1_orderkey = l1_orderkey_ptr[idx];
        bool l1_late = l1_late_ptr[idx];
        int64_t supp_id = supp_ids_ptr[idx];

        // Binary search for the orderkey in lo_orderkeys
        int64_t lo = 0;
        int64_t hi = n_lo - 1;
        int64_t mid;
        int64_t found_idx = -1;
        while (lo <= hi) {
            mid = lo + (hi - lo) / 2;
            int64_t val = lo_orderkeys_ptr[mid];
            if (val == l1_orderkey) {
                found_idx = mid;
                break;
            } else if (val < l1_orderkey) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        // If found and conditions match, add 1 to supplier count
        bool keep = false;
        if (found_idx >= 0 && l1_late && supp_id >= 0 && supp_id < num_suppliers) {
            int64_t distinct_supp = lo_distinct_supp_ptr[found_idx];
            int64_t late_distinct_supp = lo_late_distinct_supp_ptr[found_idx];
            bool exists_other = (distinct_supp >= 2);
            bool not_exists_other_late = (late_distinct_supp <= 1);
            keep = l1_late && exists_other && not_exists_other_late && (supp_id >= 0);
        }

        if (keep) {
            atomicAdd(&out_counts_ptr[supp_id], 1);
        }
    }
}

void q21_launcher(
    torch::Tensor l1_orderkey,
    torch::Tensor l1_suppkey,
    torch::Tensor l1_late,
    torch::Tensor supp_ids,
    torch::Tensor lo_orderkeys,
    torch::Tensor lo_distinct_supp,
    torch::Tensor lo_late_distinct_supp,
    int64_t num_suppliers,
    torch::Tensor out_counts
) {
    // Move all tensors to CUDA and make contiguous
    auto to_cuda_contig = [](torch::Tensor t) -> torch::Tensor {
        if (!t.is_cuda()) t = t.cuda();
        return t.contiguous();
    };
    l1_orderkey = to_cuda_contig(l1_orderkey);
    l1_suppkey = to_cuda_contig(l1_suppkey);
    l1_late = to_cuda_contig(l1_late);
    supp_ids = to_cuda_contig(supp_ids);
    lo_orderkeys = to_cuda_contig(lo_orderkeys);
    lo_distinct_supp = to_cuda_contig(lo_distinct_supp);
    lo_late_distinct_supp = to_cuda_contig(lo_late_distinct_supp);
    out_counts = to_cuda_contig(out_counts);

    int64_t n_elements = l1_orderkey.numel();
    int64_t n_lo = lo_orderkeys.numel();

    // Set out_counts to zero
    cudaMemset(out_counts.data_ptr<int>(), 0, out_counts.numel() * sizeof(int));

    int blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 0) {
        q21_kernel<<<blocks, BLOCK_SIZE>>>(
            l1_orderkey.data_ptr<int64_t>(),
            l1_suppkey.data_ptr<int64_t>(),
            l1_late.data_ptr<bool>(),
            supp_ids.data_ptr<int64_t>(),
            lo_orderkeys.data_ptr<int64_t>(),
            lo_distinct_supp.data_ptr<int64_t>(),
            lo_late_distinct_supp.data_ptr<int64_t>(),
            num_suppliers,
            out_counts.data_ptr<int>(),
            n_elements,
            n_lo
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q21_launcher", &q21_launcher, "TPC-H Q21 core (CUDA)");
}
'''

def _load_cuda():
    """Compile the CUDA kernel at import time using a temporary file."""
    tmpdir = tempfile.mkdtemp(prefix="q21_cuda_")
    cu_path = os.path.join(tmpdir, "_q21_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q21_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _launch_q21(
    l1_orderkey: torch.Tensor,
    l1_suppkey: torch.Tensor,
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    lo_orderkeys: torch.Tensor,
    lo_distinct_supp: torch.Tensor,
    lo_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch the fused CUDA kernel for Q21.
    Returns (supp_idx, counts) where supp_idx = torch.arange(num_suppliers) and counts are computed.
    """
    device = l1_orderkey.device
    # Use int32 for the output counts to match the int atomicAdd
    out_counts = torch.zeros(num_suppliers, dtype=torch.int32, device=device)
    
    # Ensure tensors have correct dtypes
    l1_orderkey = l1_orderkey.to(torch.int64)
    l1_suppkey = l1_suppkey.to(torch.int64)
    l1_late = l1_late.to(torch.bool)
    supp_ids = supp_ids.to(torch.int64)
    lo_orderkeys = lo_orderkeys.to(torch.int64)
    lo_distinct_supp = lo_distinct_supp.to(torch.int64)
    lo_late_distinct_supp = lo_late_distinct_supp.to(torch.int64)
    
    _EXT.q21_launcher(
        l1_orderkey, l1_suppkey, l1_late, supp_ids,
        lo_orderkeys, lo_distinct_supp, lo_late_distinct_supp,
        int(num_suppliers), out_counts
    )
    
    supp_idx = torch.arange(num_suppliers, device=device, dtype=torch.int64)
    # Convert counts back to int64 for the return type
    return supp_idx, out_counts.to(torch.int64)


def _query_core(
    l1_orderkey: torch.Tensor,
    l1_suppkey: torch.Tensor,
    l1_late: torch.Tensor,  # bool: receiptdate > commitdate
    supp_ids: torch.Tensor,  # per-l1 row: supplier index (into s_name list) or -1
    lo_orderkeys: torch.Tensor,        # unique orderkeys sorted
    lo_distinct_supp: torch.Tensor,    # number of distinct suppliers per orderkey
    lo_late_distinct_supp: torch.Tensor,  # number of distinct suppliers with late receipt per orderkey
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused CUDA implementation of the Q21 core logic.
    For each l1 row, checks existence conditions and atomically increments
    the count for the corresponding supplier. Returns (supp_idx, counts).
    
    This is faster than the reference because it does one fused pass over l1 rows
    with binary search per row, avoiding separate searchsorted and scatter_add_ operations.
    """
    return _launch_q21(
        l1_orderkey, l1_suppkey, l1_late, supp_ids,
        lo_orderkeys, lo_distinct_supp, lo_late_distinct_supp,
        num_suppliers,
    )


def run_query(tables: dict[str, cudf.DataFrame], nation_name: str = "SAUDI ARABIA", orderstatus: str = "F") -> cudf.DataFrame:
    """
    SQL: TPC-H Q21 - Suppliers Who Kept Orders Waiting.
    Description: For each supplier in the given nation, count orders of status 'F' where the
                 supplier's lineitem was late, another supplier also shipped the order, and no
                 other supplier was late on that order.
    Params:
      nation_name (str): nation name filter.
      orderstatus (str): order status filter.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # Filter nation -> supplier
    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supplier.merge(nation_f, left_on="s_nationkey", right_on="n_nationkey")
    supp_n = supp_n[["s_suppkey", "s_name"]].sort_values("s_name").reset_index(drop=True)
    supp_n["supp_id"] = cudf.Series(range(len(supp_n)), dtype="int32")

    # Filter orders by status
    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]

    # --- Compute per-orderkey aggregates over FULL lineitem (restricted to orders_f) ---
    li_all = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        orders_f, left_on="l_orderkey", right_on="o_orderkey"
    )
    li_all["late"] = (li_all["l_receiptdate"] > li_all["l_commitdate"]).astype("int8")

    # distinct suppliers per orderkey
    distinct_supp = (
        li_all[["l_orderkey", "l_suppkey"]]
        .drop_duplicates()
        .groupby("l_orderkey")
        .size()
        .reset_index()
    )
    distinct_supp.columns = ["l_orderkey", "distinct_supp"]

    # distinct late suppliers per orderkey
    late_rows = li_all[li_all["late"] == 1][["l_orderkey", "l_suppkey"]].drop_duplicates()
    late_distinct = late_rows.groupby("l_orderkey").size().reset_index()
    late_distinct.columns = ["l_orderkey", "late_distinct_supp"]

    agg = distinct_supp.merge(late_distinct, on="l_orderkey", how="left")
    agg["late_distinct_supp"] = agg["late_distinct_supp"].fillna(0).astype("int32")
    agg["distinct_supp"] = agg["distinct_supp"].astype("int32")
    agg = agg.sort_values("l_orderkey").reset_index(drop=True)

    # --- Build l1 rows: supplier join + orders filter ---
    l1 = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        supp_n, left_on="l_suppkey", right_on="s_suppkey"
    )
    l1 = l1.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")
    l1["late"] = (l1["l_receiptdate"] > l1["l_commitdate"]).astype("bool")

    l1_orderkey_t = torch.as_tensor(l1["l_orderkey"].to_cupy(), dtype=torch.int64)
    l1_suppkey_t = torch.as_tensor(l1["l_suppkey"].to_cupy(), dtype=torch.int64)
    l1_late_t = torch.as_tensor(l1["late"].to_cupy(), dtype=torch.bool)
    supp_ids_t = torch.as_tensor(l1["supp_id"].to_cupy(), dtype=torch.int64)

    lo_orderkeys_t = torch.as_tensor(agg["l_orderkey"].to_cupy(), dtype=torch.int64)
    lo_distinct_supp_t = torch.as_tensor(agg["distinct_supp"].to_cupy(), dtype=torch.int64)
    lo_late_distinct_supp_t = torch.as_tensor(agg["late_distinct_supp"].to_cupy(), dtype=torch.int64)

    num_suppliers = len(supp_n)

    supp_idx_t, counts_t = _query_core(
        l1_orderkey_t,
        l1_suppkey_t,
        l1_late_t,
        supp_ids_t,
        lo_orderkeys_t,
        lo_distinct_supp_t,
        lo_late_distinct_supp_t,
        num_suppliers,
    )

    result = cudf.DataFrame({
        "s_name": supp_n["s_name"].iloc[cudf.Series(supp_idx_t)].reset_index(drop=True),
        "numwait": cudf.Series(counts_t),
    })
    result = result[result["numwait"] > 0]
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100).reset_index(drop=True)
    return result
