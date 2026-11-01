# q17_triton__q17_build_index_kernel

Source: `azure__gpt-5.5/triton/full/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q17_build_index_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q17_build_index_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_build_part_index_map` launches the kernel once with grid
`(triton.cdiv(n_parts, 256),)`, `BLOCK_SIZE=256`, and `num_warps=4`.

## Lengths and ranges

- `part_keys_ptr`: the `p_partkey` values of the parts that pass the brand and
  container filter, as `int64`, `n_parts` rows. `n_parts = part_keys.numel()`;
  `run_query` and `_query_core` return before the launch when it is 0, so its
  least value is 1.
- `index_map_ptr`: `torch.full((map_size,), -1)` of `int32`, `map_size` rows,
  which the kernel only writes.
- `map_size = max(part_keys) + 1`; the plan returns before the launch when the
  largest key is negative, so its least value is 1. The plan sizes the map from
  the largest value of `part_keys_ptr`, and TPC-H keys are positive, so every
  value of `part_keys_ptr` lies in `[0, map_size - 1]`: `index-in` with target
  `index_map_ptr`.
- `n_parts`, `map_size`: Python `int`s below 2^31, `i32`.
