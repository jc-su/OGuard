# q12_triton__q12_filter_agg_kernel

Source: `azure__gpt-5.5/triton/full/queries/q12/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q12_filter_agg_kernel` of
`_KERNEL_SOURCE` (plan lines 51 to 97).

## Extraction edits

- Renamed `_q12_filter_agg_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q12_filter_agg` (plan lines 145 to 177) launches the kernel once
with grid `(triton.cdiv(n, BLOCK_SIZE),)`, `BLOCK_SIZE=_BLOCK_SIZE` (4096),
and `num_warps=8`. The plan's `_query_core` (lines 180 to 204), the
"compatibility core for the reference tensor interface", calls this launcher;
`run_query` calls `_launch_q12_counts` instead and does not call
`_query_core`.

## Lengths and ranges

- `code_ptr`, `high_ptr`, `commit_ptr`, `receipt_ptr`, `ship_ptr`: the five
  tensors of `_query_core`. The plan does not build them on this path; the
  reference `_query_core` interface of DataKernelBench's Q12 passes five int32
  columns of one joined frame (shipmode code, priority flag, and commit,
  receipt, and ship epoch days), so each has `n_elements` rows. The launcher
  takes `n = l_shipmode_code.numel()` and does not compare the other four
  lengths with it.
- `n_elements`: `i32`, size symbol with least value 1; the launcher returns
  before the launch when `n == 0`.
- `receipt_lo_days`, `receipt_hi_days`: Python `int` epoch days, `i32`;
  comparison thresholds only.
- `out_ptr`: `torch.zeros(4)` int64, added to at indices 0 to 3 with
  `tl.atomic_add` (`readwrite`).
