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
    reset_to_zero=["out_total_ptr", "out_promo_ptr"],
)
@triton.jit
def _promo_kernel(
    price_ptr, discount_ptr, shipdate_ptr, promo_ptr,
    date_lo, date_hi, out_total_ptr, out_promo_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=mask, other=0)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    revenue = price * (1.0 - discount)
    total_block = tl.sum(tl.where(keep, revenue, 0.0))
    promo_block = tl.sum(tl.where(keep & (promo > 0), revenue, 0.0))
    tl.atomic_add(out_total_ptr, total_block)
    tl.atomic_add(out_promo_ptr, promo_block)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="promo_triton_")
    path = os.path.join(tmpdir, "_promo_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_promo_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def _launch_promo(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Launches the fused Triton kernel to compute total and promo revenue.
    Returns a scalar tensor = 100.0 * promo_sum / total.
    """
    n = l_extendedprice.numel()
    out_total = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)
    out_promo = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._promo_kernel[grid](
        l_extendedprice, l_discount, l_shipdate_days, p_type_is_promo,
        shipdate_lo_days, shipdate_hi_days, out_total, out_promo, n,
    )
    total = out_total.item()
    promo = out_promo.item()
    if total == 0.0:
        return torch.tensor(0.0, dtype=torch.float64, device=l_extendedprice.device)
    return torch.tensor(100.0 * promo / total, dtype=torch.float64, device=l_extendedprice.device)


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL:
        SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
               / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue
        FROM lineitem, part
        WHERE l_partkey = p_partkey
          AND l_shipdate >= date '1995-09-01'
          AND l_shipdate < CAST('1995-10-01' AS date);

    Description:
        cuDF performs the join (lineitem × part on l_partkey = p_partkey) and string processing
        (p_type LIKE 'PROMO%') which are not easily fused in Triton. The hot path — filter on
        shipdate, project: revenue = l_extendedprice * (1 - l_discount), and aggregate total and
        promo sums — is fused into a single Triton kernel. This avoids multiple eager PyTorch
        passes and reduces HBM traffic, since the join output is scanned once.

    Params:
        tables (dict[str, cudf.DataFrame]): Dictionary containing 'lineitem' and 'part' tables.
        shipdate_lo (str, optional): Inclusive lower bound on l_shipdate. Default "1995-09-01".
        shipdate_hi (str, optional): Exclusive upper bound on l_shipdate. Default "1995-10-01".

    Returns:
        float: Percentage of revenue from promotional parts.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Precompute promo flag on part side (string ops, cuDF)
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    # Join: cuDF merge on GPU
    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    # Convert to tensors for Triton kernel
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    # Launch fused Triton kernel
    res_tensor = _launch_promo(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    return float(res_tensor.item())
