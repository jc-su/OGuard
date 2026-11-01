# q03_triton__scan_lineitem_direct_kernel

Source: `azure__gpt-5.5/triton/full/queries/q03/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_scan_lineitem_direct_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_scan_lineitem_direct_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_direct_aggregate` (plan.py line 152) launches the kernel once with grid
`(triton.cdiv(n_lineitems, 256),)`, `BLOCK_SIZE=256` (`scan_block_size`), and no
`num_warps`, so 4.

## Lengths and ranges

- `l_orderkey_ptr` (int64), `l_extendedprice_ptr`, `l_discount_ptr` (float64),
  `l_shipdate_days_ptr` (int32): lineitem columns, `n_lineitems =
  l_orderkey.numel()` rows (59,986,052 at scale factor 10, so `i32`).
- `order_flag_ptr`: the uint8 table that `_mark_orders_direct_kernel` filled,
  `capacity` elements. `l_orderkey` is a foreign key whose bound comes from
  another table, so it is `read`; the kernel bounds each key by `capacity`.
- `revenue_table_ptr`: `torch.zeros((capacity,), dtype=torch.float64)`, which the
  kernel adds to with `tl.atomic_add` (`readwrite`).
- `n_lineitems`: size symbol with least value 1 (launched only when
  `n_lineitems > 0`); `capacity`: `max(1, max eligible orderkey + 1)`, least value 1.
- `shipdate_lo_days`: Python `int` epoch day of 1995-03-15, `i32`; a comparison bound only.
