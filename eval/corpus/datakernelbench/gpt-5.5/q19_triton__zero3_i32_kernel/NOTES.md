# q19_triton__zero3_i32_kernel

Source: `azure__gpt-5.5/triton/full/queries/q19/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_zero3_i32_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_zero3_i32_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_build_part_maps` (plan.py line 225) launches the kernel once with grid
`(triton.cdiv(map_size, 1024),)`, `BLOCK_SIZE=map_block_size` (default 1024),
and no `num_warps`, so 4 warps. The launch zeroes the three dense part maps.

## Lengths and ranges

- `a_ptr`, `b_ptr`, `c_ptr`: `brand_map`, `container_map`, `size_map`, each
  `torch.empty(map_size, dtype=torch.int32)`, so `n_elements` entries; the kernel
  only stores to them (`write`).
- `n_elements`: `map_size = int(df_part["p_partkey"].max()) + 1`, 2,000,001 at
  scale factor 10, so `i32`. The function returns before the launch when `part`
  is empty, and TPC-H keys are positive, so `map_size` is at least 1.
