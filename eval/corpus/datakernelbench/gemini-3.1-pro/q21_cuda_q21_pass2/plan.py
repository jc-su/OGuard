import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

// Pass 1: Filter lineitem by 'F' orders, count per order, and pack suppkey + late flag
__global__ void q21_pass1(
    const int64_t* __restrict__ l_orderkey,
    const int64_t* __restrict__ l_suppkey,
    const int32_t* __restrict__ l_receiptdate,
    const int32_t* __restrict__ l_commitdate,
    const int32_t* __restrict__ order_map,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ lineitems,
    int64_t n_lineitems,
    int64_t max_orderkey
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lineitems) return;

    int64_t okey = l_orderkey[idx];
    if (okey > max_orderkey || okey < 0) return;

    int32_t dense_idx = order_map[okey];
    if (dense_idx >= 0) {
        // Read remaining columns only for qualifying orders to save memory bandwidth
        int32_t rdate = l_receiptdate[idx];
        int32_t cdate = l_commitdate[idx];
        bool late = rdate > cdate;
        int32_t suppkey = (int32_t)l_suppkey[idx];
        
        // Pack: suppkey in upper 31 bits, late flag in lowest bit
        uint32_t packed = (suppkey << 1) | (late ? 1 : 0);
        
        int pos = atomicAdd(&counts[dense_idx], 1);
        if (pos < 16) {
            lineitems[dense_idx * 16 + pos] = packed;
        }
    }
}

// Pass 2: Evaluate EXISTS and NOT EXISTS conditions per order, then count numwait
__global__ void q21_pass2(
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ lineitems,
    const int32_t* __restrict__ supp_map,
    int32_t* __restrict__ out_counts,
    int64_t num_orders_f,
    int64_t max_suppkey
) {
    int64_t dense_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dense_idx >= num_orders_f) return;

    int count = counts[dense_idx];
    if (count == 0) return;
    if (count > 16) count = 16; 

    int32_t items[16];
    for (int i = 0; i < count; i++) {
        items[i] = lineitems[dense_idx * 16 + i];
    }

    int u_supp[16];
    bool u_late[16];
    int u_count = 0;

    // Deduplicate suppliers and compute per-supplier late status
    for (int i = 0; i < count; i++) {
        int s = items[i] >> 1;
        bool l = items[i] & 1;
        bool found = false;
        for (int j = 0; j < u_count; j++) {
            if (u_supp[j] == s) {
                u_late[j] = u_late[j] || l;
                found = true;
                break;
            }
        }
        if (!found) {
            u_supp[u_count] = s;
            u_late[u_count] = l;
            u_count++;
        }
    }

    int late_count = 0;
    for (int j = 0; j < u_count; j++) {
        if (u_late[j]) late_count++;
    }

    // Evaluate for each original lineitem
    for (int i = 0; i < count; i++) {
        int s = items[i] >> 1;
        bool l = items[i] & 1;

        // Conditions: lineitem is late, >1 distinct suppliers, <=1 late distinct suppliers
        if (l && u_count >= 2 && late_count <= 1) {
            if (s >= 0 && s <= max_suppkey) {
                int s_id = supp_map[s];
                if (s_id != -1) {
                    atomicAdd(&out_counts[s_id], 1);
                }
            }
        }
    }
}

std::vector<torch::Tensor> q21_launcher(
    torch::Tensor l_orderkey,
    torch::Tensor l_suppkey,
    torch::Tensor l_receiptdate,
    torch::Tensor l_commitdate,
    torch::Tensor order_map,
    torch::Tensor supp_map,
    int64_t num_orders_f,
    int64_t num_supp_n
) {
    int64_t n_lineitems = l_orderkey.numel();
    int64_t max_orderkey = order_map.numel() - 1;
    int64_t max_suppkey = supp_map.numel() - 1;

    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    auto counts = torch::zeros({num_orders_f}, options);
    auto lineitems = torch::zeros({num_orders_f * 16}, options);
    auto out_counts = torch::zeros({num_supp_n}, options);

    int threads = 1024;
    int blocks1 = (n_lineitems + threads - 1) / threads;
    if (blocks1 > 0) {
        q21_pass1<<<blocks1, threads>>>(
            l_orderkey.data_ptr<int64_t>(),
            l_suppkey.data_ptr<int64_t>(),
            l_receiptdate.data_ptr<int32_t>(),
            l_commitdate.data_ptr<int32_t>(),
            order_map.data_ptr<int32_t>(),
            counts.data_ptr<int32_t>(),
            lineitems.data_ptr<int32_t>(),
            n_lineitems,
            max_orderkey
        );
    }

    int blocks2 = (num_orders_f + threads - 1) / threads;
    if (blocks2 > 0) {
        q21_pass2<<<blocks2, threads>>>(
            counts.data_ptr<int32_t>(),
            lineitems.data_ptr<int32_t>(),
            supp_map.data_ptr<int32_t>(),
            out_counts.data_ptr<int32_t>(),
            num_orders_f,
            max_suppkey
        );
    }

    return {out_counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q21_launcher", &q21_launcher, "TPC-H Q21 (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q21_cuda_")
    cu_path = os.path.join(tmpdir, "kernels.cu")
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


def run_query(tables: dict[str, cudf.DataFrame], nation_name: str = "SAUDI ARABIA", orderstatus: str = "F") -> cudf.DataFrame:
    """
    SQL: TPC-H Q21 - Suppliers Who Kept Orders Waiting.
    Description: 
      Instead of multiple cuDF merges to compute per-order statistics, we map orders and 
      suppliers to dense spaces. A single CUDA pass packs `(suppkey, late)` tuples for each 
      valid 'F' order. A second CUDA pass evaluates the `EXISTS` and `NOT EXISTS` counts 
      purely in registers, avoiding large groupby/joins and intermediate writes.
    Params:
      nation_name (str): nation name filter.
      orderstatus (str): order status filter.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # Filter nation -> supplier and assign dense IDs
    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supplier.merge(nation_f, left_on="s_nationkey", right_on="n_nationkey")
    supp_n = supp_n[["s_suppkey", "s_name"]].sort_values("s_name").reset_index(drop=True)
    
    if len(supp_n) == 0:
        return cudf.DataFrame({"s_name": cudf.Series(dtype="object"), "numwait": cudf.Series(dtype="int64")})
        
    supp_keys = torch.as_tensor(supp_n["s_suppkey"].to_cupy(), dtype=torch.int64)
    max_skey = supp_keys.max().item()
    supp_map = torch.full((max_skey + 1,), -1, dtype=torch.int32, device="cuda")
    supp_map.scatter_(0, supp_keys, torch.arange(len(supp_keys), dtype=torch.int32, device="cuda"))
    num_supp_n = len(supp_n)

    # Filter orders by status and assign dense IDs
    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]
    if len(orders_f) == 0:
        return cudf.DataFrame({"s_name": cudf.Series(dtype="object"), "numwait": cudf.Series(dtype="int64")})

    order_keys = torch.as_tensor(orders_f["o_orderkey"].to_cupy(), dtype=torch.int64)
    max_okey = order_keys.max().item()
    order_map = torch.full((max_okey + 1,), -1, dtype=torch.int32, device="cuda")
    order_map.scatter_(0, order_keys, torch.arange(len(order_keys), dtype=torch.int32, device="cuda"))
    num_orders_f = len(orders_f)

    # Lineitem tensors
    l_orderkey = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64).contiguous()
    l_suppkey = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64).contiguous()
    
    # Dates as int32 days
    l_receiptdate = torch.as_tensor(
        (df_lineitem["l_receiptdate"].astype("int64") // 86400).astype("int32").to_cupy()
    ).contiguous()
    
    l_commitdate = torch.as_tensor(
        (df_lineitem["l_commitdate"].astype("int64") // 86400).astype("int32").to_cupy()
    ).contiguous()

    # Launch fused query
    out_counts = _EXT.q21_launcher(
        l_orderkey, l_suppkey, l_receiptdate, l_commitdate,
        order_map, supp_map, num_orders_f, num_supp_n
    )[0]

    # Map back to result
    result = cudf.DataFrame({
        "s_name": supp_n["s_name"],
        "numwait": cudf.Series(out_counts).astype("int64")
    })
    
    result = result[result["numwait"] > 0]
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100).reset_index(drop=True)
    return result
