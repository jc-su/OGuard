# q16_core_triton__count_groups_from_table_kernel

Source: `azure__gpt-5.5/triton/core/queries/q16/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_count_groups_from_table_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_count_groups_from_table_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_distinct_count` launches the kernel once, when `n > 0` and
`num_groups > 0`, with grid `(triton.cdiv(table_size, 1024),)`, `BLOCK_SIZE=1024`, and
no `num_warps` (4).

## Lengths and ranges

- `table_ptr`: the `table_size` entries that `_insert_distinct_pairs_kernel` filled; read only.
- `counts_ptr`: the `num_groups` `int64` counters that `_fill_i64_kernel` zeroed, added to with `tl.atomic_add` (`readwrite`); `num_groups` is a size symbol that the kernel does not receive, least value 1.
- `table_size`: the smallest power of two of at least `max(2, 2 * n)` (`i32`): size symbol with least value 2.
