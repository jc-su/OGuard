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

@triton.jit
def _q1_filter_and_reduce(
    returnflag_ptr, linestatus_ptr, quantity_ptr, extendedprice_ptr, discount_ptr, tax_ptr, shipdate_ptr,
    shipdate_hi,
    out_qty, out_price, out_disc_price, out_charge, out_disc, out_count,
    n_elements, n_groups, n_ls,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi + 1)
    keep = mask & (shipdate <= shipdate_hi)

    returnflag = tl.load(returnflag_ptr + offs, mask=mask, other=0)
    linestatus = tl.load(linestatus_ptr + offs, mask=mask, other=0)
    group_idx = returnflag * n_ls + linestatus

    quantity = tl.load(quantity_ptr + offs, mask=keep, other=0.0)
    extendedprice = tl.load(extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    disc_price = extendedprice * (1.0 - discount)
    charge = disc_price * (1.0 + tax)

    # Accumulate per group within this block, then atomic add to global
    # Since n_groups <= 4, we iterate over each group
    for g in range(0, n_groups, 1):
        in_group = group_idx == g
        group_keep = keep & in_group
        
        sum_q = tl.sum(tl.where(group_keep, quantity, 0.0))
        sum_p = tl.sum(tl.where(group_keep, extendedprice, 0.0))
        sum_dp = tl.sum(tl.where(group_keep, disc_price, 0.0))
        sum_ch = tl.sum(tl.where(group_keep, charge, 0.0))
        sum_d = tl.sum(tl.where(group_keep, discount, 0.0))
        sum_c = tl.sum(tl.where(group_keep, 1.0, 0.0))
        
        tl.atomic_add(out_qty + g, sum_q)
        tl.atomic_add(out_price + g, sum_p)
        tl.atomic_add(out_disc_price + g, sum_dp)
        tl.atomic_add(out_charge + g, sum_ch)
        tl.atomic_add(out_disc + g, sum_d)
        tl.atomic_add(out_count + g, sum_c)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q1_triton_")
    path = os.path.join(tmpdir, "_q1_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q1_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _launch_q1(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
    n_rf: int,
    n_ls: int,
    BLOCK_SIZE: int = 4096,
) -> tuple:
    n = l_quantity.numel()
    n_groups = n_rf * n_ls

    out_qty = torch.zeros(n_groups, dtype=torch.float64, device=l_quantity.device)
    out_price = torch.zeros(n_groups, dtype=torch.float64, device=l_quantity.device)
    out_disc_price = torch.zeros(n_groups, dtype=torch.float64, device=l_quantity.device)
    out_charge = torch.zeros(n_groups, dtype=torch.float64, device=l_quantity.device)
    out_disc = torch.zeros(n_groups, dtype=torch.float64, device=l_quantity.device)
    out_count = torch.zeros(n_groups, dtype=torch.float64, device=l_quantity.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q1_filter_and_reduce[grid](
        l_returnflag_code, l_linestatus_code, l_quantity, l_extendedprice,
        l_discount, l_tax, l_shipdate_days, shipdate_hi_days,
        out_qty, out_price, out_disc_price, out_charge, out_disc, out_count,
        n, n_groups, n_ls,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    mask = out_count > 0.5
    indices = torch.nonzero(mask).squeeze(-1)
    
    rf_out = (indices // n_ls).to(torch.int64)
    ls_out = (indices % n_ls).to(torch.int64)
    
    sum_qty = out_qty[indices]
    sum_base_price = out_price[indices]
    sum_disc_price = out_disc_price[indices]
    sum_charge = out_charge[indices]
    sum_disc = out_disc[indices]
    count_order = out_count[indices].to(torch.int64)
    
    count_order_f = count_order.to(torch.float64)
    avg_qty = sum_qty / count_order_f
    avg_price = sum_base_price / count_order_f
    avg_disc = sum_disc / count_order_f

    combined = rf_out * n_ls + ls_out
    order = torch.argsort(combined)
    
    return (
        rf_out[order],
        ls_out[order],
        sum_qty[order],
        sum_base_price[order],
        sum_disc_price[order],
        sum_charge[order],
        avg_qty[order],
        avg_price[order],
        avg_disc[order],
        count_order[order],
    )

def _query_core(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """
    Runs TPC-H Q1 aggregation on GPU using a fused Triton kernel.
    """
    n_rf = int(l_returnflag_code.max().item() + 1) if l_returnflag_code.numel() > 0 else 1
    n_ls = int(l_linestatus_code.max().item() + 1) if l_linestatus_code.numel() > 0 else 1
    
    return _launch_q1(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        shipdate_hi_days,
        n_rf,
        n_ls,
    )

def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
    Description: TPC-H Q1 pricing summary report aggregated by return flag and line status.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]

    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    l_returnflag_code_t = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy())
    l_linestatus_code_t = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy())

    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    (
        rf_out, ls_out, sum_qty, sum_base_price, sum_disc_price,
        sum_charge, avg_qty, avg_price, avg_disc, count_order,
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
    )

    rf_out_host = rf_out.cpu().tolist()
    ls_out_host = ls_out.cpu().tolist()
    rf_strs = [rf_categories[i] for i in rf_out_host]
    ls_strs = [ls_categories[i] for i in ls_out_host]

    return cudf.DataFrame({
        "l_returnflag": cudf.Series(rf_strs),
        "l_linestatus": cudf.Series(ls_strs),
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": sum_disc_price,
        "sum_charge": sum_charge,
        "avg_qty": avg_qty,
        "avg_price": avg_price,
        "avg_disc": avg_disc,
        "count_order": count_order,
    })
