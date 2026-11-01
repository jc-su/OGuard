import triton
import triton.language as tl

TYPES = {
    "RF_PTR": "*i32",
    "LS_PTR": "*i32",
    "QTY_PTR": "*fp64",
    "PRICE_PTR": "*fp64",
    "DISC_PTR": "*fp64",
    "TAX_PTR": "*fp64",
    "SHIPDATE_PTR": "*i32",
    "SHIPDATE_HI": "i32",
    "OUT_SUM_QTY": "*fp64",
    "OUT_SUM_BASE": "*fp64",
    "OUT_SUM_DISC_PRICE": "*fp64",
    "OUT_SUM_CHARGE": "*fp64",
    "OUT_SUM_DISC": "*fp64",
    "OUT_COUNT": "*i32",
    "N_ELEM": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 1024}
NUM_WARPS = 4


@triton.jit
def kernel(
    # input column pointers
    RF_PTR, LS_PTR, QTY_PTR, PRICE_PTR, DISC_PTR, TAX_PTR, SHIPDATE_PTR,
    # scalar filter bound (int32 days)
    SHIPDATE_HI,
    # output per‑group accumulators (dense key = rf*256+ls)
    OUT_SUM_QTY, OUT_SUM_BASE, OUT_SUM_DISC_PRICE,
    OUT_SUM_CHARGE, OUT_SUM_DISC, OUT_COUNT,
    # meta
    N_ELEM,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM

    # Load shipdate and apply filter
    shipdate = tl.load(SHIPDATE_PTR + offs, mask=mask, other=0)
    keep = mask & (shipdate <= SHIPDATE_HI)

    # Load remaining columns only for rows that satisfy the filter
    rf = tl.load(RF_PTR + offs, mask=keep, other=0)
    ls = tl.load(LS_PTR + offs, mask=keep, other=0)
    qty = tl.load(QTY_PTR + offs, mask=keep, other=0.0)
    price = tl.load(PRICE_PTR + offs, mask=keep, other=0.0)
    disc = tl.load(DISC_PTR + offs, mask=keep, other=0.0)
    tax = tl.load(TAX_PTR + offs, mask=keep, other=0.0)

    # Dense group key (fits in int32)
    group = rf * 256 + ls

    # Derived expressions (float64 arithmetic)
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    # Atomic accumulation per group (float64 sums)
    tl.atomic_add(OUT_SUM_QTY + group, qty)
    tl.atomic_add(OUT_SUM_BASE + group, price)
    tl.atomic_add(OUT_SUM_DISC_PRICE + group, disc_price)
    tl.atomic_add(OUT_SUM_CHARGE + group, charge)
    tl.atomic_add(OUT_SUM_DISC + group, disc)

    # Conditional count increment (int32)
    inc = tl.where(keep, 1, 0)
    tl.atomic_add(OUT_COUNT + group, inc)
