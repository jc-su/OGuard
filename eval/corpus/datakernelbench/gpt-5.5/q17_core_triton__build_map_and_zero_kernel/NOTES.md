# q17_core_triton__build_map_and_zero_kernel

Source: `azure__gpt-5.5/triton/core/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_build_map_and_zero_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_build_map_and_zero_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q17_map_kernels` launches the kernel once with grid
`(triton.cdiv(n_parts, 256),)`, `BLOCK_SIZE=256`, and `num_warps=4`.

## Lengths and ranges

- `part_keys_ptr`: the filtered `p_partkey` values, `n_parts` rows (`i32`). `map_size = max(part_keys) + 1` is sized from the maximum of the same keys, and TPC-H keys are positive, so `part_keys_ptr` is `index-in` `map_ptr`.
- `map_ptr`: the `map_size` `int32` entries that `_init_map_and_out_kernel` set to -1, written only.
- `sum_qty_ptr`, `cnt_qty_ptr`: `torch.empty((n_parts,), dtype=torch.float64)` each, written only.
- `n_parts`, `map_size`: size symbols with least value 1; the launcher returns before the launches when `n_parts == 0` or `max_key < 0`.
