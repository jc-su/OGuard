# q06_core_triton__q6_kernel

Source: `azure__gpt-5.5/triton/core/queries/q06/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q6_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q6_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q6` launches the kernel once, when `n_elements > 0`, with grid
`(triton.cdiv(n_elements, 1024),)`, `BLOCK_SIZE=1024`, `num_warps=4`, and
`num_stages=4` (the front end takes no `num_stages`).

## Lengths and ranges

- `shipdate_ptr` (`int32`), `discount_ptr`, `quantity_ptr`, `price_ptr` (`float64`): the lineitem columns, `n_elements = l_extendedprice.numel()` rows each (59,986,052 at scale factor 10, so `i32`).
- `thresh_ptr`: a `torch.tensor` of the three floats `[discount_lo, discount_hi, quantity_hi]`, 3 elements.
- `out_ptr`: `torch.zeros(1)`, one element, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when it is 0.
- `date_lo`, `date_hi`: Python `int` epoch days, `i32`; comparison thresholds only.
