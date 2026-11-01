# q11_triton__q11_count_kernel

Source: `azure__gpt-5.5/triton/full/queries/q11/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q11_count_kernel` of `_KERNEL_SOURCE`
(plan lines 50 to 68).

## Extraction edits

- Renamed `_q11_count_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_extract` (plan lines 170 to 196) launches the kernel once with grid
`(triton.cdiv(n_groups, block_size),)`, `BLOCK_SIZE=block_size` (1024), and
no `num_warps` (4).

## Lengths and ranges

- `group_values_ptr`: the float64 `group_values` of `_q11_scan_kernel`,
  `max(ps_partkey) + 1` elements; `n_groups = group_values.numel()`, `i32`, a
  size symbol with least value 1.
- `total_ptr`: the 1-element float64 total of `_q11_scan_kernel`.
- `fraction_ptr`: `torch.tensor([threshold_fraction])` float64, 1 element.
- `count_ptr`: `torch.zeros(1)` int32, added to with `tl.atomic_add`
  (`readwrite`).
