# q07_core_triton__q7_core_kernel

Source: `azure__gpt-5.5/triton/core/queries/q07/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q7_core_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q7_core_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q7_core` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and no `num_warps` (4).

## Lengths and ranges

- `supp_ptr`, `cust_ptr`, `shipdate_ptr` (`int32`), `price_ptr`, `discount_ptr` (`float64`): the joined columns, `n_elements = l_shipdate_days.numel()` rows each (below 2^31, so `i32`).
- `out_ptr`: `torch.zeros(8, dtype=torch.float64)`, 8 elements, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
- `date_lo`, `date_hi`: Python `int` epoch days, `i32`; comparison thresholds only.
