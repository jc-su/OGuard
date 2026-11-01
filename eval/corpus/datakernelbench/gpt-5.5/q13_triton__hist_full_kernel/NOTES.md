# q13_triton__hist_full_kernel

Source: `azure__gpt-5.5/triton/full/queries/q13/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_hist_full_kernel` of `_KERNEL_SOURCE`
(plan lines 85 to 105).

## Extraction edits

- Renamed `_hist_full_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan lines 229 to 240) launches the kernel when the overflow
count of `_hist_partial_kernel` is not zero, that is when some customer has
256 or more counted orders, with grid `(num_customer_blocks,)` (1,465),
`BLOCK_SIZE=customer_block_size` (1024), and no `num_warps` (4).
`n_customers = c_custkey.numel()` (1,500,000 at scale factor 10),
`counts_len = max(c_custkey) + 1` (1,500,001), and
`hist_len = n_orders + 1` (15,000,001) are `tl.constexpr` arguments. TPC-H
customers have far fewer than 256 orders, so at scale factor 10 the plan does
not take this branch; the constants are the table sizes of scale factor 10.

## Lengths and ranges

With the sizes fixed by the constants, every length is a constant.

- `c_custkey_ptr`: `c_custkey` of `customer`, int64, 1,500,000 elements.
- `counts_ptr`: the int32 per-customer counts of `_count_orders_kernel`,
  1,500,001 elements; read only.
- `hist_ptr`: `hist_full = torch.zeros(hist_len)` int64, 15,000,001 elements,
  added to with `tl.atomic_add` (`readwrite`).
