# q20_core_triton__accum_lineitem_kernel

Source: `azure__gpt-5.5/triton/core/queries/q20/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `_accum_lineitem_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_accum_lineitem_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q20_triton` launches the kernel once with grid `(triton.cdiv(n_li, 256),)`,
`BLOCK_SIZE=256`, `MAX_PROBE=64`, `table_mask=ps_table_size - 1`, and `num_warps=4`.

## Lengths and ranges

- `l_partkey_ptr`, `l_suppkey_ptr` (`int64`), `l_quantity_ptr` (`float64`): the date-filtered lineitem columns, `n_lineitem` rows each (below 2^31, so `i32`); the keys are hashed only, so `read`.
- `ht_part_ptr`, `ht_supp_ptr`, `ht_idx_ptr`: the `table_mask + 1` entries that `_insert_ps_kernel` filled; read only here.
- `sum_qty_ptr`: `torch.zeros((n_ps,), dtype=torch.float64)`, added to with `tl.atomic_add` (`readwrite`); `n_ps` is a size symbol that the kernel does not receive, least value 1.
- `n_lineitem`: size symbol with least value 1; `table_mask`: a power of two minus one of at least 15 (`i32`).
