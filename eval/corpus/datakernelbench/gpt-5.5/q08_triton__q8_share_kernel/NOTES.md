# q08_triton__q8_share_kernel

Source: `azure__gpt-5.5/triton/full/queries/q08/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q8_share_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q8_share_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q8_aggregate` (plan.py line 152) launches the kernel once, after
`_q8_agg_kernel`, with grid `(triton.cdiv(num_years, share_bs),)`,
`BLOCK_SIZE=share_bs`, and no `num_warps`, so 4. `_share_block_size` returns 16
for `num_years <= 16`; the order years of the query lie in 1995 and 1996, so
`num_years` is at most 2 and `BLOCK_SIZE=16`.

## Lengths and ranges

- `target_ptr`, `total_ptr`: `torch.zeros(num_years, dtype=torch.float64)`,
  filled by `_q8_agg_kernel`; `share_ptr`: `torch.empty(num_years)`, which the
  kernel only stores into (`write`). All three have `n_years` elements.
- `n_years`: `num_years = max_year - base_year + 1`, a Python `int`, `i32`; size
  symbol with least value 1 (the plan returns before the launch when
  `num_years <= 0`).
