# q14_core_triton__q14_finalize_kernel

Source: `azure__gpt-5.5/triton/core/queries/q14/round_04.py` (`output_round` 4),
copied as `plan.py`. The kernel is `_q14_finalize_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q14_finalize_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q14` launches the kernel once with grid `(1,)`,
`BLOCK_SIZE=_next_power_of_2(max(n_blocks, 1))` (8192 at scale factor 10:
`n_blocks = triton.cdiv(59986052, 8192) = 7323`), `partial_stride=max(n_blocks, 1)`,
and `num_warps=4`.

## Lengths and ranges

- `partials_ptr`: the `2 * partial_stride` partial sums of `_q14_scan_kernel`; read only.
- `out_ptr`: `torch.empty(())`, a 0-dim tensor of one element, written only.
- `partial_stride`: `max(n_blocks, 1)`, size symbol with least value 1 (`i32`). `n_blocks`: the program count of the scan, a Python `int` in `[0, 2^31)` (`i32`), the load bound.
