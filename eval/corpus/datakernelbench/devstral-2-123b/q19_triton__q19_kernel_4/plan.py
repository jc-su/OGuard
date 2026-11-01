import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q19_kernel(
    l_quantity_ptr, l_extendedprice_ptr, l_discount_ptr,
    p_brand_id_ptr, p_container_grp_ptr, p_size_ptr,
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load filter columns
    p_brand_id = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    p_container_grp = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    p_size = tl.load(p_size_ptr + offs, mask=mask, other=0)
    l_quantity = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    l_extendedprice = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    l_discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)

    # Compute conditions
    cond1 = (
        (p_brand_id == brand12_id)
        & (p_container_grp == sm_grp)
        & (l_quantity >= 1.0)
        & (l_quantity <= 11.0)
        & (p_size >= 1)
        & (p_size <= 5)
    )
    cond2 = (
        (p_brand_id == brand23_id)
        & (p_container_grp == md_grp)
        & (l_quantity >= 10.0)
        & (l_quantity <= 20.0)
        & (p_size >= 1)
        & (p_size <= 10)
    )
    cond3 = (
        (p_brand_id == brand34_id)
        & (p_container_grp == lg_grp)
        & (l_quantity >= 20.0)
        & (l_quantity <= 30.0)
        & (p_size >= 1)
        & (p_size <= 15)
    )
    keep = mask & (cond1 | cond2 | cond3)

    # Compute revenue and sum
    revenue = tl.where(keep, l_extendedprice * (1.0 - l_discount), 0.0)
    block_sum = tl.sum(revenue)
    tl.atomic_add(out_ptr, block_sum)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q19_triton_")
    path = os.path.join(tmpdir, "_q19_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q19_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(
    tables: dict[str, cudf.DataFrame],
    brand1: str = "Brand#12",
    brand2: str = "Brand#23",
    brand3: str = "Brand#34",
    shipinstruct: str = "DELIVER IN PERSON",
) -> float:
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]
    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part[part_mask]

    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    brand12_id, brand23_id, brand34_id = 12, 23, 34
    b12 = ((joined["p_brand"] == brand1)).astype("int32") * brand12_id
    b23 = ((joined["p_brand"] == brand2)).astype("int32") * brand23_id
    b34 = ((joined["p_brand"] == brand3)).astype("int32") * brand34_id
    p_brand_id = b12 + b23 + b34

    sm_grp, md_grp, lg_grp = 1, 2, 3
    sm_mask = (
        (joined["p_container"] == "SM CASE") | (joined["p_container"] == "SM BOX")
        | (joined["p_container"] == "SM PACK") | (joined["p_container"] == "SM PKG")
    ).astype("int32") * sm_grp
    md_mask = (
        (joined["p_container"] == "MED BAG") | (joined["p_container"] == "MED BOX")
        | (joined["p_container"] == "MED PKG") | (joined["p_container"] == "MED PACK")
    ).astype("int32") * md_grp
    lg_mask = (
        (joined["p_container"] == "LG CASE") | (joined["p_container"] == "LG BOX")
        | (joined["p_container"] == "LG PACK") | (joined["p_container"] == "LG PKG")
    ).astype("int32") * lg_grp
    container_grp = sm_mask + md_mask + lg_mask

    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(p_brand_id.to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(container_grp.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    n = l_quantity_t.numel()
    out = torch.zeros(1, dtype=torch.float64, device=l_quantity_t.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q19_kernel[grid](
        l_quantity_t, l_extendedprice_t, l_discount_t,
        p_brand_id_t, p_container_grp_t, p_size_t,
        brand12_id, brand23_id, brand34_id,
        sm_grp, md_grp, lg_grp,
        out, n,
    )

    return float(out.item())
