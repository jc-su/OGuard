# q15_triton__q15_agg_kernel

Source: `azure__gpt-5.5/triton/full/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q15_agg_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q15_agg_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, when `n_line > 0`, with grid
`(triton.cdiv(n_line, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=8`. `n_line`,
`n_sup`, and `BINARY_STEPS` are `tl.constexpr` arguments:

- `n_line = l_suppkey.numel()`, the rows of `lineitem`: 59,986,052 at scale
  factor 10 (the benchmark's `meta_data/sf10/summarize/lineitem.csv`).
- `n_sup = s_suppkey.numel()`, the rows of `supplier`: 100,000.
- `BINARY_STEPS = max(1, (n_sup - 1).bit_length() + 1) = 18`.

## Lengths and ranges

- `l_suppkey_ptr` (`int64`), `l_price_ptr`, `l_discount_ptr` (`float64`),
  `l_shipdate_days_ptr` (`int32`): the `lineitem` columns, 59986052 rows each.
- `sorted_suppkey_ptr`: `torch.sort(s_suppkey)`, 100000 `int64` values.
- `totals_ptr`: 100000 `float64`, which the kernel adds to with `tl.atomic_add`
  (`readwrite`); `present_ptr`: 100000 `uint8`, which the kernel only writes.
  The plan's init kernel zeroes both before this launch.
- `shipdate_lo_days`, `shipdate_hi_days`: Python `int` epoch days, `i32`;
  comparison thresholds only.
- The lengths are constants because `n_line` and `n_sup` are constexprs of
  this configuration.
