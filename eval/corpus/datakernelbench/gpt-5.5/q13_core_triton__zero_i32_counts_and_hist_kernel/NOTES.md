# q13_core_triton__zero_i32_counts_and_hist_kernel

Source: `azure__gpt-5.5/triton/core/queries/q13/round_06.py` (`output_round` 6),
copied as `plan.py`. The kernel is `_zero_i32_counts_and_hist_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_zero_i32_counts_and_hist_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once with grid
`(ceil(max(counts_n, 64) / 1024),)`, `HIST_BINS=64`, `BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `counts_ptr`: `scratch32[:counts_n]`, the first `counts_n` `int32` entries of the scratch buffer, written only.
- `hist_ptr`: `scratch32[counts_n:counts_n + 64]`, the next 64 entries, written only; the plan passes it as its own tensor of 64 elements.
- `counts_n`: `num_customers + 1` with `num_customers = max(c_custkey)` (1,500,001 at scale factor 10, so `i32`): size symbol with least value 1.
