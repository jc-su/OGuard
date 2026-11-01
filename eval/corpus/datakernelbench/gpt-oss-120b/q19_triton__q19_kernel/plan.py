"""
Optimized TPC-H Q19 using Triton fusion.

SQL:
SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem, part
WHERE (p_partkey = l_partkey
  AND p_brand = 'Brand#12'
  AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
  AND l_quantity BETWEEN 1 AND 11
  AND p_size BETWEEN 1 AND 5
  AND l_shipmode IN ('AIR','AIR REG')
  AND l_shipinstruct = 'DELIVER IN PERSON')
 OR (p_partkey = l_partkey
  AND p_brand = 'Brand#23'
  AND p_container IN ('MED BAG','MED BOX','MED PKG','MED PACK')
  AND l_quantity BETWEEN 10 AND 20
  AND p_size BETWEEN 1 AND 10
  AND l_shipmode IN ('AIR','AIR REG')
  AND l_shipinstruct = 'DELIVER IN PERSON')
 OR (p_partkey = l_partkey
  AND p_brand = 'Brand#34'
  AND p_container IN ('LG CASE','LG BOX','LG PACK','LG PKG')
  AND l_quantity BETWEEN 20 AND 30
  AND p_size BETWEEN 1 AND 15
  AND l_shipmode IN ('AIR','AIR REG')
  AND l_shipinstruct = 'DELIVER IN PERSON');

Description:
  * cuDF handles string filters and the join.
  * Brand and container are encoded as small int32 columns on the GPU.
  * All numeric work (three disjunction predicates, revenue projection,
    global sum) is fused into a single Triton kernel, removing multiple
    HBM passes and temporaries.  The kernel performs a vectorized scan,
    builds a per‑lane mask, and uses an atomic add to a single fp64
    accumulator — ideal for the memory‑bound nature of this query on
    H100.

Params:
    tables (dict[str, cudf.DataFrame]): mapping with keys "lineitem" and "part".
    brand1, brand2, brand3 (str): brand literals (default = TPC‑H values).
    shipinstruct (str): ship instruction literal (default = "DELIVER IN PERSON").

Returns:
    float: the aggregated revenue.
"""

import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton  # needed for grid computation (triton.cdiv)

# ----------------------------------------------------------------------
# Triton kernel source – written to a temporary file and imported at load time
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotuned configurations for H100 (SM90)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],   # required because we use atomic_add
)
@triton.jit
def _q19_kernel(
    qty_ptr,                # float64 *
    extprice_ptr,           # float64 *
    disc_ptr,               # float64 *
    brand_id_ptr,           # int32 *
    cont_grp_ptr,           # int32 *
    size_ptr,               # int32 *
    brand12_id, brand23_id, brand34_id,   # int32 scalars
    sm_grp, md_grp, lg_grp,                # int32 scalars
    out_ptr,                # float64 *
    n_elements,             # int64
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all needed columns (masked)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    ext = tl.load(extprice_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    brand = tl.load(brand_id_ptr + offs, mask=mask, other=0)
    cont = tl.load(cont_grp_ptr + offs, mask=mask, other=0)
    size = tl.load(size_ptr + offs, mask=mask, other=0)

    # Three predicate groups
    cond1 = (brand == brand12_id) & (cont == sm_grp) & \
            (qty >= 1.0) & (qty <= 11.0) & (size >= 1) & (size <= 5)
    cond2 = (brand == brand23_id) & (cont == md_grp) & \
            (qty >= 10.0) & (qty <= 20.0) & (size >= 1) & (size <= 10)
    cond3 = (brand == brand34_id) & (cont == lg_grp) & \
            (qty >= 20.0) & (qty <= 30.0) & (size >= 1) & (size <= 15)

    keep = cond1 | cond2 | cond3

    # Revenue = l_extendedprice * (1 - l_discount)
    revenue = ext * (1.0 - disc)

    # Accumulate only rows that satisfy the combined mask
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, revenue, 0.0)))
'''

def _load_kernels():
    """Write the Triton kernel source to a temporary Python file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="q19_triton_")
    path = os.path.join(tmpdir, "q19_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q19_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels once at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Launch helper that wraps the Triton kernel
# ----------------------------------------------------------------------
def _launch_q19(
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    p_brand_id: torch.Tensor,
    p_container_grp: torch.Tensor,
    p_size: torch.Tensor,
    brand12_id: int,
    brand23_id: int,
    brand34_id: int,
    sm_grp: int,
    md_grp: int,
    lg_grp: int,
) -> torch.Tensor:
    """Execute the fused Triton kernel and return a 0‑dim fp64 tensor."""
    n = l_quantity.numel()
    out = torch.zeros(1, dtype=torch.float64, device=l_quantity.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q19_kernel[grid](
        l_quantity,
        l_extendedprice,
        l_discount,
        p_brand_id,
        p_container_grp,
        p_size,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
        out,
        n,
    )
    return out.squeeze(0)

# ----------------------------------------------------------------------
# Core wrapper – thin pass‑through to the launch helper
# ----------------------------------------------------------------------
def _query_core(
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    p_brand_id: torch.Tensor,
    p_container_grp: torch.Tensor,
    p_size: torch.Tensor,
    brand12_id: int,
    brand23_id: int,
    brand34_id: int,
    sm_grp: int,
    md_grp: int,
    lg_grp: int,
) -> torch.Tensor:
    """
    Calls the fused Triton kernel that evaluates the three disjunctions,
    computes revenue, and returns the global sum as a scalar tensor.
    """
    return _launch_q19(
        l_quantity,
        l_extendedprice,
        l_discount,
        p_brand_id,
        p_container_grp,
        p_size,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
    )

# ----------------------------------------------------------------------
# Public API – unchanged signature from the reference implementation
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    brand1: str = "Brand#12",
    brand2: str = "Brand#23",
    brand3: str = "Brand#34",
    shipinstruct: str = "DELIVER IN PERSON",
) -> float:
    """
    SQL: SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE
         (p_partkey=l_partkey AND p_brand='Brand#12' AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
          AND l_quantity BETWEEN 1 AND 11 AND p_size BETWEEN 1 AND 5
          AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct='DELIVER IN PERSON')
         OR (... Brand#23, MED containers, qty 10-20, size 1-10 ...)
         OR (... Brand#34, LG containers, qty 20-30, size 1-15 ...).
    Description: TPC‑H Q19 – uses cuDF for string‑heavy filtering & join, then a
                 single Triton kernel for the numeric scan and aggregation.
    Params:
        tables       – mapping containing "lineitem" and "part" DataFrames.
        brand1/2/3  – brand literals (default TPC‑H values).
        shipinstruct – ship instruction literal (default "DELIVER IN PERSON").
    Returns:
        float – the computed revenue.
    """
    # ------------------------------------------------------------------
    # 1. cuDF preprocessing: string filters + inner join
    # ------------------------------------------------------------------
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Shipmode & instruction common to all disjunctions
    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]

    # Brand filter on part (common narrowing)
    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part[part_mask]

    # Join on primary key
    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    # ------------------------------------------------------------------
    # 2. Encode brand and container to small int32 groups
    # ------------------------------------------------------------------
    brand12_id, brand23_id, brand34_id = 12, 23, 34
    b12 = (joined["p_brand"] == brand1).astype("int32") * brand12_id
    b23 = (joined["p_brand"] == brand2).astype("int32") * brand23_id
    b34 = (joined["p_brand"] == brand3).astype("int32") * brand34_id
    p_brand_id = b12 + b23 + b34  # int32 series

    sm_grp, md_grp, lg_grp = 1, 2, 3
    sm_mask = (
        (joined["p_container"] == "SM CASE")
        | (joined["p_container"] == "SM BOX")
        | (joined["p_container"] == "SM PACK")
        | (joined["p_container"] == "SM PKG")
    ).astype("int32") * sm_grp
    md_mask = (
        (joined["p_container"] == "MED BAG")
        | (joined["p_container"] == "MED BOX")
        | (joined["p_container"] == "MED PKG")
        | (joined["p_container"] == "MED PACK")
    ).astype("int32") * md_grp
    lg_mask = (
        (joined["p_container"] == "LG CASE")
        | (joined["p_container"] == "LG BOX")
        | (joined["p_container"] == "LG PACK")
        | (joined["p_container"] == "LG PKG")
    ).astype("int32") * lg_grp
    p_container_grp = sm_mask + md_mask + lg_mask  # int32 series

    # ------------------------------------------------------------------
    # 3. Materialize numeric columns as torch tensors (fp64 / int32)
    # ------------------------------------------------------------------
    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(p_brand_id.to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(p_container_grp.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    # ------------------------------------------------------------------
    # 4. Execute fused GPU kernel
    # ------------------------------------------------------------------
    result_tensor = _query_core(
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        p_brand_id_t,
        p_container_grp_t,
        p_size_t,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
    )
    return float(result_tensor.item())
