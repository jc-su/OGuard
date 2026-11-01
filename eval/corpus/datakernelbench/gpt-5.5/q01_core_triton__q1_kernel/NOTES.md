# q01_core_triton__q1_kernel

Source: `azure__gpt-5.5/triton/core/queries/q01/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q1_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q1_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q1` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 4096),)`, `BLOCK_SIZE=4096`, and `num_warps=8`.

## Lengths and ranges

- `rf_ptr`, `ls_ptr`, `shipdate_ptr` (`int32`), `qty_ptr`, `price_ptr`, `disc_ptr`, `tax_ptr` (`float64`): the lineitem columns, `n_elements = l_quantity.numel()` rows each (59,986,052 at scale factor 10, so `i32`).
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
- `sum_qty_ptr`, `sum_base_price_ptr`, `sum_disc_price_ptr`, `sum_charge_ptr`, `sum_disc_ptr`, `count_ptr`: `torch.zeros(16, dtype=torch.float64)` each, 16 elements, added to with `tl.atomic_add` (`readwrite`).
- `shipdate_hi_days`: a Python `int` epoch day, `i32`; a comparison threshold only.
