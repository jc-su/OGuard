# q18_triton__sum_quantity_by_orderkey_kernel

Source: `azure__gpt-5.5/triton/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_sum_quantity_by_orderkey_kernel` of
`_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_sum_quantity_by_orderkey_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, when `n_lineitem > 0`, with grid
`(triton.cdiv(n_lineitem, 256),)`, `BLOCK_SIZE=sum_block_size=256`, and
`num_warps=4`.

## Lengths and ranges

- `orderkey_ptr` (`int64`), `quantity_ptr` (`float64`): the `lineitem` columns,
  `n_elements` rows; `n_elements = l_orderkey.numel()`, least value 1, `i32`.
- `sum_by_key_ptr`: `torch.empty(n_slots)` of `float64`, zeroed by the plan's
  fill kernel, which this kernel adds to with `tl.atomic_add` (`readwrite`).
  `n_slots = max_orderkey + 1`, where `run_query` computes `max_orderkey` as
  the larger of the largest `o_orderkey` and the largest `l_orderkey`. Every
  value of `orderkey_ptr` is therefore at most `n_slots - 1`, and TPC-H keys are
  positive: `index-in` with target `sum_by_key_ptr`. Least value of `n_slots`: 1.
