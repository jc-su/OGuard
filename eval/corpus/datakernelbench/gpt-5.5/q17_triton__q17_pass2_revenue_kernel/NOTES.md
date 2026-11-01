# q17_triton__q17_pass2_revenue_kernel

Source: `azure__gpt-5.5/triton/full/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q17_pass2_revenue_kernel` of
`_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q17_pass2_revenue_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, after the first pass, with grid
`(triton.cdiv(n_rows, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=8`.

## Lengths and ranges

- `l_partkey_ptr` (`int64`), `l_quantity_ptr`, `l_extendedprice_ptr`
  (`float64`): the `lineitem` columns, `n_rows` rows; `n_rows` least value 1,
  `i32`.
- `index_map_ptr`: the `int32` map of the build kernel, `map_size` rows, least
  value 1, `i32`; values -1 or a filtered-part ordinal, role `read`.
- `sum_qty_ptr`, `cnt_qty_ptr`: the `float64` sums and counts of the first
  pass, `n_parts` rows (least value 1), read.
- `out_ptr`: `torch.zeros((1,))`, one `float64`, which the kernel adds to with
  `tl.atomic_add` (`readwrite`).
