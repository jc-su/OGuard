import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf

# -------------------------------------------------------------------------
# Triton kernels for TPC‑H Q17 (Level: core)
# -------------------------------------------------------------------------
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
    reset_to_zero=["sum_qty_ptr", "cnt_qty_ptr"],
)
@triton.jit
def _q17_phase1_kernel(
    part_idx_ptr,      # int64 per‑row index into sorted part keys
    mask_ptr,          # uint8 mask indicating part‑membership
    qty_ptr,           # float64 quantity
    sum_qty_ptr,       # float64 output: sum of quantities per part
    cnt_qty_ptr,       # float64 output: count of rows per part
    n_elements: int,   # number of lineitem rows
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load membership mask, part index, and quantity
    mask_vals = tl.load(mask_ptr + offs, mask=mask, other=0).to(tl.int1)
    part_idx = tl.load(part_idx_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)

    # Compute contributions only for rows that belong to filtered parts
    # Use a loop to do per-element atomic adds (simple and correct)
    for i in range(BLOCK_SIZE):
        global_off = pid * BLOCK_SIZE + i
        if global_off < n_elements:
            m = tl.load(mask_ptr + global_off).to(tl.int1)
            if m:
                p = tl.load(part_idx_ptr + global_off)
                q = tl.load(qty_ptr + global_off)
                tl.atomic_add(sum_qty_ptr + p, q)
                tl.atomic_add(cnt_qty_ptr + p, 1.0)


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
def _q17_phase2_kernel(
    part_idx_ptr,      # int64 per‑row index into sorted part keys
    mask_ptr,          # uint8 mask indicating part‑membership
    qty_ptr,           # float64 quantity
    price_ptr,         # float64 extended price
    threshold_ptr,     # float64 per‑part threshold (0.2 * avg qty)
    out_ptr,           # float64 output accumulator
    n_elements: int,   # number of lineitem rows
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Process each element individually with atomic adds
    for i in range(BLOCK_SIZE):
        global_off = pid * BLOCK_SIZE + i
        if global_off < n_elements:
            mask_val = tl.load(mask_ptr + global_off).to(tl.int1)
            if mask_val:
                p = tl.load(part_idx_ptr + global_off)
                q = tl.load(qty_ptr + global_off)
                price = tl.load(price_ptr + global_off)
                thresh = tl.load(threshold_ptr + p)
                if q < thresh:
                    tl.atomic_add(out_ptr, price)
'''

def _load_kernels():
    """Write kernel source to a temporary file and load it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="q17_triton_")
    path = os.path.join(tmpdir, "_q17_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q17_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load Triton kernels at import time
_K = _load_kernels()


def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC-H Q17: sum(l_extendedprice)/7.0 where l_quantity < 0.2 * avg(l_quantity) over parts
    matching brand/container filters, grouped by partkey.
    """
    # -----------------------------------------------------------------
    # 1. Build mapping from lineitem rows to filtered part indices
    # -----------------------------------------------------------------
    sorted_keys, _ = torch.sort(part_keys)
    n_parts = sorted_keys.numel()

    # Edge case: no matching parts → result is zero
    if n_parts == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=l_partkey.device)

    n_elements = l_partkey.numel()

    # Searchsorted gives the insertion position in sorted_keys
    idx = torch.searchsorted(sorted_keys, l_partkey)
    idx_clamped = torch.clamp(idx, max=n_parts - 1)

    # Membership test: row belongs to a filtered part iff the key matches
    in_part = sorted_keys[idx_clamped] == l_partkey

    # Convert boolean mask to uint8 for the Triton kernels
    mask_u8 = in_part.to(torch.uint8)

    # Per‑row index into the sorted part array (meaningful only when in_part is True)
    part_idx = idx_clamped

    # -----------------------------------------------------------------
    # 2. Phase 1: accumulate quantity and count per part key
    # -----------------------------------------------------------------
    sum_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_partkey.device)
    cnt_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_partkey.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _K._q17_phase1_kernel[grid](
        part_idx, mask_u8, l_quantity,
        sum_qty, cnt_qty,
        n_elements,
    )

    # -----------------------------------------------------------------
    # 3. Compute per‑part threshold = 0.2 * avg(quantity)
    # -----------------------------------------------------------------
    avg_qty = torch.where(cnt_qty > 0, sum_qty / cnt_qty, torch.zeros_like(sum_qty))
    threshold = 0.2 * avg_qty

    # -----------------------------------------------------------------
    # 4. Phase 2: compute the final sum of extendedprice for rows that satisfy the predicate
    # -----------------------------------------------------------------
    out = torch.zeros(1, dtype=torch.float64, device=l_partkey.device)
    _K._q17_phase2_kernel[grid](
        part_idx, mask_u8, l_quantity, l_extendedprice,
        threshold, out, n_elements,
    )

    # Return the required result: sum / 7.0
    return out.squeeze(0) / 7.0


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                            FROM lineitem
                            WHERE l_partkey = p_partkey);
    Description: TPC-H Q17. Computes the small-quantity order revenue average for parts of a
        given brand and container. Correlated subquery is evaluated as a grouped average over
        all lineitems for each filtered partkey.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table first
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
