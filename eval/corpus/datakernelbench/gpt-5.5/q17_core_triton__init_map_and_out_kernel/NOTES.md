# q17_core_triton__init_map_and_out_kernel

Source: `azure__gpt-5.5/triton/core/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_init_map_and_out_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_init_map_and_out_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q17_map_kernels` launches the kernel once, when `n_lineitems > 0`,
`n_parts > 0`, and `max_key >= 0`, with grid `(triton.cdiv(map_size, 1024),)`,
`BLOCK_SIZE=1024`, and `num_warps=8`.

## Lengths and ranges

- `map_ptr`: `torch.empty((map_size,), dtype=torch.int32)`, `n_elements = map_size = max(part_keys) + 1` elements (at most 2,000,001, so `i32`), written only.
- `out_ptr`: `torch.empty((1,), dtype=torch.float64)`, one element, written only.
- `n_elements`: size symbol with least value 1.
