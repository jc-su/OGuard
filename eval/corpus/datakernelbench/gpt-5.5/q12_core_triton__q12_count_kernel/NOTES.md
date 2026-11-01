# q12_core_triton__q12_count_kernel

Source: `azure__gpt-5.5/triton/core/queries/q12/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q12_count_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q12_count_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q12_count_kernel` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `shipmode_code_ptr`, `priority_high_ptr`, `commitdate_ptr`, `receiptdate_ptr`, `shipdate_ptr` (`int32`): the joined columns, `n_elements = l_shipmode_code.numel()` rows each (below 2^31, so `i32`).
- `high_out_ptr`, `low_out_ptr`: `torch.zeros(2, dtype=torch.int64)` each, 2 elements, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
- `receipt_lo_days`, `receipt_hi_days`: Python `int` epoch days, `i32`; comparison thresholds only.
