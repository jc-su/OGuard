import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused kernel implementing the core of TPC‑H Q21.
# One pass over the filtered l1 rows performs:
#   * binary‑search into per‑order aggregates,
#   * EXISTS / NOT‑EXISTS predicate evaluation,
#   * per‑supplier histogram update via block‑level atomics.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

// ---------------------------------------------------------------
// One thread processes a single l1 row.
// ---------------------------------------------------------------
__global__ void q21_kernel(
    const int64_t* __restrict__ l1_orderkey,
    const unsigned char* __restrict__ l1_late,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int64_t* __restrict__ lo_distinct_supp,
    const int64_t* __restrict__ lo_late_distinct_supp,
    unsigned long long* __restrict__ counts,
    const int64_t n,
    const int64_t lo_n)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // ---- load l1 row -------------------------------------------------
    const int64_t okey   = l1_orderkey[idx];
    const unsigned char late = l1_late[idx];
    const int64_t supp_id = supp_ids[idx];

    // ---- binary search (lower_bound) into lo_orderkeys -----------------
    int64_t lo = 0;
    int64_t hi = lo_n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t mid_key = lo_orderkeys[mid];
        if (mid_key < okey)
            lo = mid + 1;
        else
            hi = mid;
    }
    const int64_t pos = lo;   // exact match guaranteed by preprocessing

    // ---- retrieve aggregates for this orderkey ------------------------
    const int64_t distinct = lo_distinct_supp[pos];
    const int64_t late_distinct = lo_late_distinct_supp[pos];

    // ---- evaluate predicates (identical to reference logic) ----------
    const bool exists_other = (distinct >= 2);
    const bool not_exists_other_late = (late_distinct <= 1);
    const bool mask = (late != 0) && exists_other && not_exists_other_late && (supp_id >= 0);

    // ---- atomic increment of per‑supplier counter --------------------
    if (mask) {
        atomicAdd(&counts[supp_id], 1ULL);
    }
}

// ---------------------------------------------------------------
// Launcher called from Python.
// ---------------------------------------------------------------
void q21_launcher(
    torch::Tensor l1_orderkey,
    torch::Tensor l1_late,               // must be uint8
    torch::Tensor supp_ids,
    torch::Tensor lo_orderkeys,
    torch::Tensor lo_distinct_supp,
    torch::Tensor lo_late_distinct_supp,
    torch::Tensor counts)
{
    // ----- type checks -----
    TORCH_CHECK(l1_orderkey.is_cuda(), "l1_orderkey must be CUDA");
    TORCH_CHECK(l1_late.is_cuda(),   "l1_late must be CUDA");
    TORCH_CHECK(supp_ids.is_cuda(),  "supp_ids must be CUDA");
    TORCH_CHECK(lo_orderkeys.is_cuda(),   "lo_orderkeys must be CUDA");
    TORCH_CHECK(lo_distinct_supp.is_cuda(), "lo_distinct_supp must be CUDA");
    TORCH_CHECK(lo_late_distinct_supp.is_cuda(), "lo_late_distinct_supp must be CUDA");
    TORCH_CHECK(counts.is_cuda(),    "counts must be CUDA");

    TORCH_CHECK(l1_orderkey.dtype() == torch::kInt64, "l1_orderkey dtype");
    TORCH_CHECK(l1_late.dtype() == torch::kUInt8,    "l1_late dtype must be uint8");
    TORCH_CHECK(supp_ids.dtype() == torch::kInt64,   "supp_ids dtype");
    TORCH_CHECK(lo_orderkeys.dtype() == torch::kInt64,   "lo_orderkeys dtype");
    TORCH_CHECK(lo_distinct_supp.dtype() == torch::kInt64, "lo_distinct_supp dtype");
    TORCH_CHECK(lo_late_distinct_supp.dtype() == torch::kInt64,
                "lo_late_distinct_supp dtype");
    TORCH_CHECK(counts.dtype() == torch::kInt64, "counts dtype");

    // ----- ensure contiguity -----
    l1_orderkey = l1_orderkey.contiguous();
    l1_late = l1_late.contiguous();
    supp_ids = supp_ids.contiguous();
    lo_orderkeys = lo_orderkeys.contiguous();
    lo_distinct_supp = lo_distinct_supp.contiguous();
    lo_late_distinct_supp = lo_late_distinct_supp.contiguous();
    counts = counts.contiguous();

    const int64_t n = l1_orderkey.size(0);
    const int64_t lo_n = lo_orderkeys.size(0);

    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    q21_kernel<<<blocks, threads>>>(
        l1_orderkey.data_ptr<int64_t>(),
        reinterpret_cast<const unsigned char*>(l1_late.data_ptr<uint8_t>()),
        supp_ids.data_ptr<int64_t>(),
        lo_orderkeys.data_ptr<int64_t>(),
        lo_distinct_supp.data_ptr<int64_t>(),
        lo_late_distinct_supp.data_ptr<int64_t>(),
        reinterpret_cast<unsigned long long*>(counts.data_ptr<int64_t>()),
        n,
        lo_n
    );

    // propagate any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q21_launcher", &q21_launcher,
          "TPC‑H Q21 fused core kernel (CUDA)");
}
'''

def _load_cuda():
    """Compile the CUDA source into a PyTorch extension (once per import)."""
    tmpdir = tempfile.mkdtemp(prefix="q21_cuda_")
    cu_path = os.path.join(tmpdir, "q21_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q21_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        verbose=False,
        build_directory=build_dir,
    )

# Load (and compile) the extension
_EXT = _load_cuda()


def _launch_q21_core(
    l1_orderkey: torch.Tensor,
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    lo_orderkeys: torch.Tensor,
    lo_distinct_supp: torch.Tensor,
    lo_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
) -> torch.Tensor:
    """
    Prepare the output buffer and invoke the fused CUDA kernel.
    Returns a 1‑D int64 CUDA tensor with per‑supplier counts.
    """
    # Ensure l1_late is uint8 as required by the kernel
    l1_late_uint8 = l1_late.to(torch.uint8)

    counts = torch.zeros(num_suppliers,
                        dtype=torch.int64,
                        device=l1_orderkey.device)

    _EXT.q21_launcher(
        l1_orderkey,
        l1_late_uint8,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        counts,
    )
    return counts


def _query_core(
    l1_orderkey: torch.Tensor,
    l1_suppkey: torch.Tensor,               # kept for API compatibility (unused)
    l1_late: torch.Tensor,
    supp_ids: torch.Tensor,
    lo_orderkeys: torch.Tensor,
    lo_distinct_supp: torch.Tensor,
    lo_late_distinct_supp: torch.Tensor,
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core computation for TPC‑H Q21.
    Replaces the eager‑tensor version with a single fused CUDA kernel that:
      • binary‑searches per‑order aggregates,
      • evaluates EXISTS / NOT‑EXISTS predicates,
      • increments a per‑supplier histogram via block‑level atomics.
    Returns (supplier_index_tensor, count_tensor) matching the reference API.
    """
    counts = _launch_q21_core(
        l1_orderkey,
        l1_late,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        num_suppliers,
    )
    supp_idx = torch.arange(num_suppliers,
                            device=counts.device,
                            dtype=torch.int64)
    return supp_idx, counts


def run_query(tables: dict[str, cudf.DataFrame],
              nation_name: str = "SAUDI ARABIA",
              orderstatus: str = "F") -> cudf.DataFrame:
    """
    SQL: TPC‑H Q21 - Suppliers Who Kept Orders Waiting.
    Description: For each supplier in the given nation, count orders of status 'F'
    where the supplier's lineitem was late, another supplier also shipped the order,
    and no other supplier was late on that order.
    Params:
      nation_name (str): nation name filter.
      orderstatus (str): order status filter.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # ---- nation → supplier filtering ----
    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supplier.merge(nation_f, left_on="s_nationkey", right_on="n_nationkey")
    supp_n = supp_n[["s_suppkey", "s_name"]].sort_values("s_name").reset_index(drop=True)
    supp_n["supp_id"] = cudf.Series(range(len(supp_n)), dtype="int32")

    # ---- orders status filter ----
    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]

    # ---- per‑order aggregates over the full lineitem (restricted to orders_f) ----
    li_all = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        orders_f, left_on="l_orderkey", right_on="o_orderkey"
    )
    li_all["late"] = (li_all["l_receiptdate"] > li_all["l_commitdate"]).astype("int8")

    distinct_supp = (
        li_all[["l_orderkey", "l_suppkey"]]
        .drop_duplicates()
        .groupby("l_orderkey")
        .size()
        .reset_index()
    )
    distinct_supp.columns = ["l_orderkey", "distinct_supp"]

    late_rows = li_all[li_all["late"] == 1][["l_orderkey", "l_suppkey"]].drop_duplicates()
    late_distinct = late_rows.groupby("l_orderkey").size().reset_index()
    late_distinct.columns = ["l_orderkey", "late_distinct_supp"]

    agg = distinct_supp.merge(late_distinct, on="l_orderkey", how="left")
    agg["late_distinct_supp"] = agg["late_distinct_supp"].fillna(0).astype("int32")
    agg["distinct_supp"] = agg["distinct_supp"].astype("int32")
    agg = agg.sort_values("l_orderkey").reset_index(drop=True)

    # ---- build l1 rows (supplier join + orders filter) ----
    l1 = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        supp_n, left_on="l_suppkey", right_on="s_suppkey"
    )
    l1 = l1.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")
    l1["late"] = (l1["l_receiptdate"] > l1["l_commitdate"]).astype("bool")

    l1_orderkey_t = torch.as_tensor(l1["l_orderkey"].to_cupy(), dtype=torch.int64)
    l1_suppkey_t = torch.as_tensor(l1["l_suppkey"].to_cupy(), dtype=torch.int64)  # unused
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
    result = result.sort_values(["numwait", "s_name"],
                                ascending=[False, True]).head(100).reset_index(drop=True)
    return result
