# q14_core_triton__q14_scan_kernel

Source: `azure__gpt-5.5/triton/core/queries/q14/round_04.py` (`output_round` 4),
copied as `plan.py`. The kernel is `_q14_scan_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q14_scan_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q14` launches the kernel once, when `n_blocks > 0`, with grid `(n_blocks,)`,
`n_blocks = triton.cdiv(n, 8192)`, `BLOCK_SIZE=8192`, `partial_stride=n_blocks`, and
`num_warps=8`.

## Lengths and ranges

- `price_ptr`, `discount_ptr` (`float64`), `shipdate_ptr` (`int32`), `promo_flag_ptr` (a `torch.bool` tensor, `*i1`): the joined columns, `n_elements = l_extendedprice.numel()` rows each (below 2^31, so `i32`).
- `partials_ptr`: `torch.empty((partial_stride * 2,), dtype=torch.float64)`, `2 * partial_stride` elements, written only.
- `partial_stride`: `max(n_blocks, 1)` with `n_blocks = triton.cdiv(n, 8192)`, equal to `n_blocks` at this launch: size symbol with least value 1 (`i32`).
- `n_elements`: size symbol with least value 1. `date_lo`, `date_hi`: `i32` comparison thresholds only.
