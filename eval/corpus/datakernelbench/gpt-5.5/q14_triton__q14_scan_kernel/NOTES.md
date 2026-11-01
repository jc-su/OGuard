# q14_triton__q14_scan_kernel

Source: `azure__gpt-5.5/triton/full/queries/q14/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q14_scan_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q14_scan_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q14_scan` launches the kernel once with grid
`(triton.cdiv(n, BLOCK_SIZE),)`, `BLOCK_SIZE=4096` (the default `block_size`
of `run_query`), and `num_warps=8`.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `shipdate_days_ptr`, `promo_ptr`: `n_elements`
  rows each, the columns of the lineitem and part join that `run_query` builds
  with cuDF; `n_elements` is `l_extendedprice.numel()`, below 2^31 at scale
  factor 10, so `i32`. `promo_ptr` is a `torch.bool` tensor (`*i1`).
- `n_elements`: size symbol with least value 1; `_launch_q14_scan` returns
  before the launch when `n == 0`.
- `out_ptr`: `torch.zeros(2)`, two elements, which the kernel adds to with
  `tl.atomic_add` at `out_ptr` and `out_ptr + 1` (`readwrite`).
- `date_lo`, `date_hi`: Python `int` epoch days, `i32`; comparison thresholds
  only.
