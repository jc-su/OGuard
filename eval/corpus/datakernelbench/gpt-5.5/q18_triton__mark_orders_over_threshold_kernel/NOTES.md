# q18_triton__mark_orders_over_threshold_kernel

Source: `azure__gpt-5.5/triton/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_mark_orders_over_threshold_kernel` of
`_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_mark_orders_over_threshold_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, when `n_orders > 0`, with grid
`(triton.cdiv(n_orders, 1024),)`, `BLOCK_SIZE=scan_block_size=1024`, and
`num_warps=4`.

## Lengths and ranges

- `orderkey_ptr`: the `o_orderkey` column as `int64`, `n_elements` rows;
  `n_elements = o_orderkey.numel()`, least value 1, `i32`. `max_orderkey`
  includes the largest `o_orderkey`, so every value is at most `n_slots - 1`,
  and TPC-H keys are positive: `index-in` with target `sum_by_key_ptr`.
- `sum_by_key_ptr`: the `float64` table of `n_slots = max_orderkey + 1` rows
  that the sum kernel filled, read. Least value of `n_slots`: 1.
- `threshold_ptr`: `torch.tensor([qty_threshold])`, one `float64`.
- `out_sum_ptr` (`float64`), `out_keep_ptr` (`torch.bool`, `*i1`):
  `torch.empty(n_orders)`, `n_elements` rows each, which the kernel only writes.
