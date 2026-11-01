import triton
import triton.language as tl

TYPES = {
    "rf_ptr": "*i32",
    "ls_ptr": "*i32",
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "tax_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "date_hi": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"N_RF": 3, "N_LS": 2, "BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    rf_ptr,
    ls_ptr,
    qty_ptr,
    price_ptr,
    disc_ptr,
    tax_ptr,
    shipdate_ptr,
    date_hi,
    out_ptr,
    n_elements,
    N_RF: tl.constexpr,
    N_LS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=valid, other=date_hi + 1)
    keep = valid & (shipdate <= date_hi)

    rf = tl.load(rf_ptr + offs, mask=keep, other=-1)
    ls = tl.load(ls_ptr + offs, mask=keep, other=-1)

    qty = tl.load(qty_ptr + offs, mask=keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)
    one64 = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    zero64 = tl.full((BLOCK_SIZE,), 0.0, tl.float64)

    # out layout per group: [sum_qty, sum_base_price, sum_disc_price,
    #                        sum_charge, sum_disc, count_as_fp64, reserved]
    for r in tl.static_range(0, N_RF):
        for s in tl.static_range(0, N_LS):
            gmask = keep & (rf == r) & (ls == s)
            base = (r * N_LS + s) * 7

            sum_qty = tl.sum(tl.where(gmask, qty, zero64))
            sum_base = tl.sum(tl.where(gmask, price, zero64))
            sum_disc_price = tl.sum(tl.where(gmask, disc_price, zero64))
            sum_charge = tl.sum(tl.where(gmask, charge, zero64))
            sum_disc = tl.sum(tl.where(gmask, disc, zero64))
            cnt = tl.sum(tl.where(gmask, one64, zero64))

            tl.atomic_add(out_ptr + base + 0, sum_qty, sem="relaxed")
            tl.atomic_add(out_ptr + base + 1, sum_base, sem="relaxed")
            tl.atomic_add(out_ptr + base + 2, sum_disc_price, sem="relaxed")
            tl.atomic_add(out_ptr + base + 3, sum_charge, sem="relaxed")
            tl.atomic_add(out_ptr + base + 4, sum_disc, sem="relaxed")
            tl.atomic_add(out_ptr + base + 5, cnt, sem="relaxed")
