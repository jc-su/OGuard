# q19_triton__build_part_maps_kernel

Source: `azure__gpt-5.5/triton/full/queries/q19/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_build_part_maps_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_build_part_maps_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_build_part_maps` (plan.py line 234) launches the kernel once, after
`_zero3_i32_kernel`, with grid `(triton.cdiv(n_parts, 1024),)`,
`BLOCK_SIZE=map_block_size` (default 1024), and no `num_warps`, so 4 warps.

## Lengths and ranges

- `p_partkey_ptr` (`int64`), `p_brand_id_ptr`, `p_container_grp_ptr`, `p_size_ptr`
  (`int32`): columns of `part` or computed from them by cuDF, `n_parts` rows each;
  `n_parts = len(df_part)`, 2,000,000 at scale factor 10, at least 1 because the
  function returns before the launch for an empty `part`.
- `brand_map_ptr`, `container_map_ptr`, `size_map_ptr`: `torch.empty(map_size)`
  of `int32`, `map_size` entries each; the kernel only stores to them (`write`).
- `map_size`: `int(df_part["p_partkey"].max()) + 1`, at least 1.
- `p_partkey_ptr` is `index-in` into `brand_map_ptr`: the plan sizes the maps from
  the maximum of the same column, and TPC-H keys are positive, so every value lies
  in `[0, map_size - 1]`. The kernel also checks `0 <= key < map_size` before
  each store.
