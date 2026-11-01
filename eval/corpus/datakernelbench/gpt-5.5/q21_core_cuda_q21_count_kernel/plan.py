import os
import tempfile
from typing import Any

import torch
import cudf
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void q21_count_kernel(
    const int64_t* __restrict__ l1_orderkey,
    const bool* __restrict__ l1_late,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int64_t* __restrict__ lo_distinct_supp,
    const int64_t* __restrict__ lo_late_distinct_supp,
    int64_t* __restrict__ counts,
    int64_t n_l1,
    int64_t n_orders,
    int64_t num_suppliers
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = tid; i < n_l1; i += stride) {
        int64_t sid = supp_ids[i];

        // Reference mask starts with l1_late and supp_ids >= 0.  Check these
        // first so non-candidate rows avoid the random binary-search loads.
        if (!l1_late[i] || sid < 0 || sid >= num_suppliers) {
            continue;
        }

        int64_t key = l1_orderkey[i];

        // lower_bound equivalent to torch.searchsorted(lo_orderkeys, key).
        int64_t lo = 0;
        int64_t hi = n_orders;
        while (lo < hi) {
            int64_t mid = lo + ((hi - lo) >> 1);
            int64_t v = lo_orderkeys[mid];
            if (v < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // The reference data construction guarantees a matching aggregate row
        // for every l1 order.  The equality guard prevents an illegal load if
        // malformed input is supplied.
        if (lo >= n_orders || lo_orderkeys[lo] != key) {
            continue;
        }

        int64_t distinct_supp = lo_distinct_supp[lo];
        int64_t late_distinct_supp = lo_late_distinct_supp[lo];

        // EXISTS another supplier and NOT EXISTS another late supplier.
        if (distinct_supp >= 2 && late_distinct_supp <= 1) {
            atomicAdd(
                reinterpret_cast<unsigned long long int*>(counts + sid),
                static_cast<unsigned long long int>(1)
            );
        }
    }
}

torch::Tensor q21_count_launcher(
    torch::Tensor l1_orderkey,
    torch::Tensor l1_late,
    torch::Tensor supp_ids,
    torch::Tensor lo_orderkeys,
    torch::Tensor lo_distinct_supp,
    torch::Tensor lo_late_distinct_supp,
    int64_t num_suppliers
) {
    if (!l1_orderkey.is_cuda()) l1_orderkey = l1_orderkey.cuda();
    if (!l1_late.is_cuda()) l1_late = l1_late.cuda();
    if (!supp_ids.is_cuda()) supp_ids = supp_ids.cuda();
    if (!lo_orderkeys.is_cuda()) lo_orderkeys = lo_orderkeys.cuda();
    if (!lo_distinct_supp.is_cuda()) lo_distinct_supp = lo_distinct_supp.cuda();
    if (!lo_late_distinct_supp.is_cuda()) lo_late_distinct_supp = lo_late_distinct_supp.cuda();

    l1_orderkey = l1_orderkey.contiguous();
    l1_late = l1_late.contiguous();
    supp_ids = supp_ids.contiguous();
    lo_orderkeys = lo_orderkeys.contiguous();
    lo_distinct_supp = lo_distinct_supp.contiguous();
    lo_late_distinct_supp = lo_late_distinct_supp.contiguous();

    TORCH_CHECK(l1_orderkey.scalar_type() == torch::kInt64, "l1_orderkey must be int64");
    TORCH_CHECK(l1_late.scalar_type() == torch::kBool, "l1_late must be bool");
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64, "supp_ids must be int64");
    TORCH_CHECK(lo_orderkeys.scalar_type() == torch::kInt64, "lo_orderkeys must be int64");
    TORCH_CHECK(lo_distinct_supp.scalar_type() == torch::kInt64, "lo_distinct_supp must be int64");
    TORCH_CHECK(lo_late_distinct_supp.scalar_type() == torch::kInt64, "lo_late_distinct_supp must be int64");
    TORCH_CHECK(num_suppliers >= 0, "num_suppliers must be non-negative");

    int64_t n_l1 = l1_orderkey.numel();
    int64_t n_orders = lo_orderkeys.numel();

    TORCH_CHECK(l1_late.numel() == n_l1, "l1_late length mismatch");
    TORCH_CHECK(supp_ids.numel() == n_l1, "supp_ids length mismatch");
    TORCH_CHECK(lo_distinct_supp.numel() == n_orders, "lo_distinct_supp length mismatch");
    TORCH_CHECK(lo_late_distinct_supp.numel() == n_orders, "lo_late_distinct_supp length mismatch");

    auto counts = torch::zeros({num_suppliers}, l1_orderkey.options().dtype(torch::kInt64));

    if (n_l1 > 0 && n_orders > 0 && num_suppliers > 0) {
        int64_t blocks64 = (n_l1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        q21_count_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            l1_orderkey.data_ptr<int64_t>(),
            l1_late.data_ptr<bool>(),
            supp_ids.data_ptr<int64_t>(),
            lo_orderkeys.data_ptr<int64_t>(),
            lo_distinct_supp.data_ptr<int64_t>(),
            lo_late_distinct_supp.data_ptr<int64_t>(),
            counts.data_ptr<int64_t>(),
            n_l1,
            n_orders,
            num_suppliers
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q21_count_launcher", &q21_count_launcher, "TPC-H Q21 fused search/filter/count CUDA kernel");
}
'''


def _load_cuda() -> Any:
    """
    Compile the CUDA hot-path extension from an in-module .cu string.

    The cuDF joins/filters remain in run_query, while the tensor hot path runs
    on the GPU in one fused CUDA kernel.  This avoids the reference sequence of
    torch.searchsorted, temporary masks, temporary selected supplier ids, ones
    allocation, and scatter_add, replacing them with one pass over l1 rows and
    direct atomic increments for qualifying rows.
    """
    tmpdir = tempfile.mkdtemp(prefix="q21_cuda_")
    cu_path = os.path.join(tmpdir, "_q21_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q21_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _query_core(
    l1_orderkey: torch.Tensor,
    l1_suppkey: torch.Tensor,
    l1_late: torch.Tensor,  # bool: receiptdate > commitdate
    supp_ids: torch.Tensor,  # per-l1 row: supplier index (into s_name list) or -1
    # Aggregated per-orderkey stats from full lineitem (l2/l3 semantics)
    lo_orderkeys: torch.Tensor,        # unique orderkeys sorted
    lo_distinct_supp: torch.Tensor,    # number of distinct suppliers per orderkey
    lo_late_distinct_supp: torch.Tensor,  # number of distinct suppliers with late receipt per orderkey
    num_suppliers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU tensor hot path for TPC-H Q21.

    cuDF work in run_query prepares compact l1 rows and per-order aggregates.
    This function then runs a hand-written CUDA kernel that fuses the reference
    searchsorted -> predicate mask -> supplier projection -> count aggregation
    into a single GPU pass.  The query is mostly memory-bound with irregular
    orderkey lookups; checking l1_late before binary search and avoiding large
    temporary tensors reduces HBM traffic and PyTorch kernel-launch overhead.
    """
    counts = _EXT.q21_count_launcher(
        l1_orderkey,
        l1_late,
        supp_ids,
        lo_orderkeys,
        lo_distinct_supp,
        lo_late_distinct_supp,
        int(num_suppliers),
    )
    return torch.arange(num_suppliers, device=counts.device), counts


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
