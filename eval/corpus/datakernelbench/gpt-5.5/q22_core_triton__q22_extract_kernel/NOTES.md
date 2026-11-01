# q22_core_triton__q22_extract_kernel

Source: `azure__gpt-5.5/triton/core/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q22_extract_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q22_extract_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q22` launches the kernel once with grid `(1,)`, `BLOCK_SIZE=128`, and
`num_warps=4`.

## Lengths and ranges

- `counts_ptr` (`int64`), `sums_ptr` (`float64`): the 100 accumulators of `_q22_agg_kernel`; read only.
- `codes_out_ptr` (`int32`), `counts_out_ptr` (`int64`), `sums_out_ptr` (`float64`): `torch.empty(100)` each, 100 elements, written only; the plan slices the first `num_groups` afterwards.
- `num_groups_ptr`: `torch.empty(1, dtype=torch.int32)`, one element, written only.
