# q03_triton__mark_orders_direct_kernel

Source: `azure__gpt-5.5/triton/full/queries/q03/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_mark_orders_direct_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_mark_orders_direct_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_direct_aggregate` (plan.py line 137) launches the kernel once with grid
`(triton.cdiv(n_orders, 256),)`, `BLOCK_SIZE=256` (`mark_block_size`), and no
`num_warps`, so 4.

## Lengths and ranges

- `o_orderkey_ptr` (int64), `o_orderdate_days_ptr` (int32),
  `o_shippriority_ptr` (int32): the columns of `eligible_orders` (orders of
  BUILDING customers before 1995-03-15), `n_orders = o_orderkey.numel()` rows.
- `order_flag_ptr` (`torch.zeros`, uint8), `orderdate_table_ptr` and
  `shippriority_table_ptr` (`torch.empty`, int32): `capacity` elements each, which
  the kernel only stores into (`write`).
- `capacity`: `max(1, max(eligible o_orderkey) + 1)`, computed from the same
  `o_orderkey` values, which are positive, so every key lies in
  `[0, capacity - 1]`: `o_orderkey_ptr` is `index-in` `order_flag_ptr` (the
  three tables have the same length). At most 60,000,001 at scale factor 10, so `i32`.
- `n_orders`: size symbol with least value 1 (launched only when `n_orders > 0`);
  `capacity`: least value 1.
