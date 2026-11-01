# q04_core_triton__mark_late_orders_kernel

Source: `azure__gpt-5.5/triton/core/queries/q04/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_mark_late_orders_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_mark_late_orders_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q4_triton` launches the kernel once, when `n_line > 0`, with grid
`(triton.cdiv(n_line, 1024),)`, `BLOCK_SIZE=1024`, and no `num_warps` (4).

## Lengths and ranges

- `l_orderkey_ptr` (`int64`), `l_commitdate_days_ptr`, `l_receiptdate_days_ptr` (`int32`): the lineitem columns, `n_elements = l_orderkey.numel()` rows each (59,986,052 at scale factor 10, so `i32`).
- `flags_ptr`: the `flag_n` bytes that `_zero_u8_kernel` cleared, written only.
- `flag_n`: `max(l_orderkey) + 1` from the same keys (`_triton_max_i64`), at least 1: size symbol with least value 1. TPC-H keys are positive, so `l_orderkey_ptr` is `index-in` `flags_ptr`.
- `n_elements`: size symbol with least value 1; the launch runs only when `n_line > 0`.
