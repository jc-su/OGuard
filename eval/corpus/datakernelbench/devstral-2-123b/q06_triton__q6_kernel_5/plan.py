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
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q6_kernel(
    price_ptr, discount_ptr, quantity_ptr, shipdate_ptr,
    date_lo, date_hi, thresh_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi = tl.load(thresh_ptr + 2)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=qty_hi)
    keep = (
        mask
        & (shipdate >= date_lo) & (shipdate < date_hi)
        & (discount >= disc_lo) & (discount <= disc_hi)
        & (quantity < qty_hi)
    )
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, price * discount, 0.0)))
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="qXX_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
    discount_lo: float = 0.05,
    discount_hi: float = 0.07,
    quantity_hi: float = 24.0,
) -> float:
    """
    SQL: SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem
         WHERE l_shipdate >= CAST('1994-01-01' AS date) AND l_shipdate < CAST('1995-01-01' AS date)
         AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;
    Description: Computes the forecast revenue increase from eliminating discounts in a given range for a year.
                 Optimized Triton kernel using a single fused pass for filtering and reduction, reducing memory-bound overhead.
    Params:
        shipdate_lo (str): inclusive lower bound ship date.
        shipdate_hi (str): exclusive upper bound ship date.
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
    """
    df_lineitem = tables["lineitem"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_shipdate_days = ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    l_discount = df_lineitem["l_discount"].to_cupy()
    l_quantity = df_lineitem["l_quantity"].to_cupy()
    l_extendedprice = df_lineitem["l_extendedprice"].to_cupy()

    device = torch.device("cuda")
    price_t = torch.as_tensor(l_extendedprice, dtype=torch.float64, device=device)
    discount_t = torch.as_tensor(l_discount, dtype=torch.float64, device=device)
    quantity_t = torch.as_tensor(l_quantity, dtype=torch.float64, device=device)
    shipdate_t = torch.as_tensor(l_shipdate_days, dtype=torch.int32, device=device)

    n = price_t.numel()
    out = torch.zeros(1, dtype=torch.float64, device=device)
    thresh_t = torch.tensor([discount_lo, discount_hi, quantity_hi], dtype=torch.float64, device=device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q6_kernel[grid](
        price_t, discount_t, quantity_t, shipdate_t,
        shipdate_lo_days, shipdate_hi_days, thresh_t, out, n,
    )
    return float(out.item())
