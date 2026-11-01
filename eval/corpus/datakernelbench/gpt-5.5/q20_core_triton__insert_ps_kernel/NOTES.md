# q20_core_triton__insert_ps_kernel

Source: `azure__gpt-5.5/triton/core/queries/q20/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `_insert_ps_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_insert_ps_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q20_triton` launches the kernel once, when `n_ps`, `n_li`, and `n_s` are all
positive, with grid `(n_ps,)`, one program per filtered partsupp row,
`MAX_PROBE=64`, `table_mask=ps_table_size - 1`, and `num_warps=1`.

## Lengths and ranges

- `ps_partkey_ptr`, `ps_suppkey_ptr`: the filtered partsupp keys, `n_ps` rows each; hashed only, so `read`. `n_ps` is a size symbol that the kernel does not receive, least value 1.
- `ht_part_ptr`, `ht_supp_ptr`: `torch.empty((ps_table_size,), dtype=torch.int64)`, written only; `ht_idx_ptr`: `torch.full((ps_table_size,), -1)`, probed with `tl.atomic_cas` (`readwrite`); `ps_table_size = table_mask + 1` entries each.
- `table_mask`: `ps_table_size - 1` with `ps_table_size` the smallest power of two of at least `max(16, 8 * n_ps)` (`i32`): size symbol with least value 15.
