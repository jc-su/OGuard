# q18_triton__fill_zero_f64_kernel

Source: `azure__gpt-5.5/triton/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_fill_zero_f64_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_fill_zero_f64_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_triton_zero_f64` launches the kernel once, for `sum_by_key`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=scan_block_size=1024`, and
`num_warps=4`.

## Lengths and ranges

- `out_ptr`: `sum_by_key = torch.empty(n_slots)` of `float64`, which the kernel
  only writes; `n_elements = out.numel()`.
- `n_elements`: size symbol with least value 1; `_triton_zero_f64` returns
  before the launch when `n == 0`. At scale factor 10 it is the largest
  orderkey plus one, 60,000,001, below 2^31, so `i32`.
