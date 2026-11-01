# q13_core_triton__sort_fixed_hist_i32_kernel

Source: `azure__gpt-5.5/triton/core/queries/q13/round_06.py` (`output_round` 6),
copied as `plan.py`. The kernel is `_sort_fixed_hist_i32_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_sort_fixed_hist_i32_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once with grid `(1,)`, `HIST_BINS=64`, and
`num_warps=2`.

## Lengths and ranges

- `hist_ptr`: the 64 `int32` bins of the histogram kernels; read only.
- `out_counts_ptr`, `out_dists_ptr`: `torch.empty(64, dtype=torch.int64)` each, 64 elements, written only; the plan slices the first `n_unique` afterwards.
- `n_unique_ptr`: `torch.empty(1, dtype=torch.int64)`, one element, written only.
