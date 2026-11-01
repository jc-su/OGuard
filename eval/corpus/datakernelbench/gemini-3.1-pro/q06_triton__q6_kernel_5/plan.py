import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

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
    # Load FP64 thresholds from memory to prevent PyTorch float -> FP32 drift
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi = tl.load(thresh_ptr + 2)
    
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Conditional loading logic to save memory bandwidth: 
    # we progressively restrict the mask so we only load subsequent columns 
    # if the previous predicates are met.
    
    # 1. Filter on shipdate
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    
    # 2. Filter on discount
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    keep = keep & (discount >= disc_lo) & (discount <= disc_hi)
    
    # 3. Filter on quantity
    quantity = tl.load(quantity_ptr + offs, mask=keep, other=qty_hi)
    keep = keep & (quantity < qty_hi)
    
    # 4. Load price only for matches, then multiply and accumulate
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, price * discount, 0.0)))
'''

def _load_kernels():
    """
    Writes the Triton kernel source to disk and loads it dynamically.
    This bypasses potential extraction errors where Triton requires true on-disk sources.
    """
    tmpdir = tempfile.mkdtemp(prefix="q6_triton_")
    path = os.path.join(tmpdir, "_q6_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q6_kernels", path)
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
                 The workload is entirely memory-bound (simple reduction). This implementation uses a single,
                 fused Triton kernel with conditional loads to bypass unneeded column fetches in low-selectivity data.
    Params:
        shipdate_lo (str): inclusive lower bound ship date.
        shipdate_hi (str): exclusive upper bound ship date.
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
    """
    df_lineitem = tables["lineitem"]

    # Calculate integer day thresholds on host
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Tensorize columns directly from cupy underlying arrays (zero-copy when compatible)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    n = l_extendedprice_t.numel()
    if n == 0:
        return 0.0

    # Output accumulator tensor
    out = torch.zeros(1, dtype=torch.float64, device=l_extendedprice_t.device)
    
    # Store FP64 scalar thresholds in an array to load faithfully from HBM
    thresh = torch.tensor(
        [discount_lo, discount_hi, quantity_hi],
        dtype=torch.float64, device=l_extendedprice_t.device,
    )

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q6_kernel[grid](
        l_extendedprice_t, l_discount_t, l_quantity_t, l_shipdate_days_t,
        shipdate_lo_days, shipdate_hi_days, thresh, out, n,
    )
    
    return float(out.item())
