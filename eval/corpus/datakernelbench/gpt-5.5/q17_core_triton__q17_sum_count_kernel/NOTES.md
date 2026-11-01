# q17_core_triton__q17_sum_count_kernel

Source: `azure__gpt-5.5/triton/core/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q17_sum_count_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q17_sum_count_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q17_map_kernels` launches the kernel once with grid
`(triton.cdiv(n_lineitems, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=8`.

## Lengths and ranges

- `l_partkey_ptr` (`int64`), `l_quantity_ptr` (`float64`): the lineitem columns, `n_lineitems` rows each (59,986,052 at scale factor 10, so `i32`); `l_partkey_ptr` is a foreign key bounded by the kernel, so `read`.
- `map_ptr`: the `map_size` entries of `_build_map_and_zero_kernel`; read only.
- `sum_qty_ptr`, `cnt_qty_ptr`: the `n_parts` accumulators that `_build_map_and_zero_kernel` zeroed, added to with `tl.atomic_add` (`readwrite`); `n_parts` is a size symbol that the kernel does not receive, least value 1.
- `n_lineitems`, `map_size`: size symbols with least value 1.
