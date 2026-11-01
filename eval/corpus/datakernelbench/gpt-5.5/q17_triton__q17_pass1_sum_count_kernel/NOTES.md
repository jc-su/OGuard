# q17_triton__q17_pass1_sum_count_kernel

Source: `azure__gpt-5.5/triton/full/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q17_pass1_sum_count_kernel` of
`_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q17_pass1_sum_count_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once with grid `(triton.cdiv(n_rows, 1024),)`,
`BLOCK_SIZE=1024`, and `num_warps=8`.

## Lengths and ranges

- `l_partkey_ptr` (`int64`), `l_quantity_ptr` (`float64`): the `lineitem`
  columns, `n_rows` rows; `n_rows = l_partkey.numel()`, least value 1
  (`_query_core` returns before the launch when it is 0), `i32`.
- `index_map_ptr`: the `int32` map that the build kernel filled, `map_size`
  rows; `map_size = index_map.numel()`, least value 1 (the plan returns when it
  is 0), `i32`. Its values are -1 or a filtered-part ordinal below `n_parts`,
  which an earlier kernel wrote, so the role is `read`. `l_partkey_ptr` holds
  keys of all parts, which can exceed `map_size - 1`; the kernel checks
  `0 <= partkey < map_size`, and the role is `read`.
- `sum_qty_ptr`, `cnt_qty_ptr`: `torch.zeros((n_parts,))` of `float64`,
  `n_parts` rows (the filtered parts, least value 1), which the kernel adds to
  with `tl.atomic_add` (`readwrite`).
