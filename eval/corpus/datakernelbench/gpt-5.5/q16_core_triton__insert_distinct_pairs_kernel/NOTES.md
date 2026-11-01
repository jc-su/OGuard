# q16_core_triton__insert_distinct_pairs_kernel

Source: `azure__gpt-5.5/triton/core/queries/q16/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_insert_distinct_pairs_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_insert_distinct_pairs_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_distinct_count` launches the kernel once, when `n > 0` and
`num_groups > 0`, with grid `(triton.cdiv(n, 256),)`, `BLOCK_SIZE=256`
(`insert_block_size`), `table_mask=table_size - 1`, and no `num_warps` (4).

## Lengths and ranges

- `group_ptr`, `supp_ptr`: the joined `_gid` and `ps_suppkey` columns, `n_elements = group_ids.numel()` rows each (below 2^31, so `i32`); both are hashed only, so `read`.
- `table_ptr`: the `table_size = table_mask + 1` `int64` entries that `_fill_i64_kernel` set to -1, probed with `tl.atomic_cas` (`readwrite`).
- `table_mask`: `table_size - 1` with `table_size` the smallest power of two of at least `max(2, 2 * n)` (`i32`): size symbol with least value 1.
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
