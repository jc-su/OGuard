# q06_triton__q6_fused_filter_project_sum_kernel

Source: `azure__gpt-5.5/triton/full/queries/q06/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q6_fused_filter_project_sum_kernel` of
`_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q6_fused_filter_project_sum_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q6_triton` launches the kernel once with grid
`(triton.cdiv(n_elements, 2048),)`, `BLOCK_SIZE=2048`, `num_warps=8`, and
`num_stages=4` (the front end takes no `num_stages`).

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `quantity_ptr`, `shipdate_days_ptr`:
  `n_elements` rows each, the columns of `lineitem`; `n_elements` is
  `l_extendedprice.numel()`, below 2^31 at scale factor 10, so `i32`.
- `n_elements`: size symbol with least value 1; the launcher returns before the
  launch when `n_elements == 0`.
- `thresholds_ptr`: a `torch.tensor` of the three floats
  `[discount_lo, discount_hi, quantity_hi]`, 3 elements.
- `out_ptr`: `torch.zeros(1)`, one element, which the kernel adds to with
  `tl.atomic_add` (`readwrite`).
- `date_lo`, `date_hi`: Python `int` epoch days, `i32`; comparison thresholds
  only.
