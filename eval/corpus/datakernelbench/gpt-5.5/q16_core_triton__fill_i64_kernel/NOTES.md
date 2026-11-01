# q16_core_triton__fill_i64_kernel

Source: `azure__gpt-5.5/triton/core/queries/q16/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_fill_i64_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_fill_i64_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_fill_i64` launches the kernel once per call, when `n > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and no `num_warps` (4).
`_launch_distinct_count` calls it twice with this configuration, for `counts` with
value 0 and for `table` with value -1; both launches are this corpus kernel.

## Lengths and ranges

- `out_ptr`: `counts` (`num_groups` elements) or `table` (`table_size` elements), `n_elements = out.numel()` `int64` elements, written only.
- `n_elements`: size symbol with least value 1; `_fill_i64` returns before the launch when `n == 0`. Below 2^31 at scale factor 10, so `i32`.
- `value`: a Python `int` (0 or -1), `i32`; stored only.
