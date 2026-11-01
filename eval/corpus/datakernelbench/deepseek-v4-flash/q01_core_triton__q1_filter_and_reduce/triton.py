import triton
import triton.language as tl

TYPES = {
    "returnflag_ptr": "*i32",
    "linestatus_ptr": "*i32",
    "quantity_ptr": "*fp64",
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "tax_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "shipdate_hi": "i32",
    "out_qty": "*fp64",
    "out_price": "*fp64",
    "out_disc_price": "*fp64",
    "out_charge": "*fp64",
    "out_disc": "*fp64",
    "out_count": "*fp64",
    "n_elements": "i32",
    "n_groups": "i32",
    "n_ls": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
}

NUM_WARPS = 4


@triton.jit
def kernel(
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
