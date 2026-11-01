# q02_core_triton__q2_limit_arange_kernel

Source: `azure__gpt-5.5/triton/core/queries/q02/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q2_limit_arange_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q2_limit_arange_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_limit_arange` launches the kernel once, when `k > 0`, with grid
`(triton.cdiv(k, 1024),)`, `n_elements=k` as a `tl.constexpr`, `BLOCK_SIZE=1024`, and
no `num_warps` (4). `k = min(limit, n)` with the default `limit` of 100 and `n` the
rows of the sorted Q2 result, which exceed 100 at scale factor 10 (the query's
answer has 100 rows), so `n_elements` is 100.

## Lengths and ranges

- `out_ptr`: `torch.empty((k,), dtype=torch.int64)`, 100 elements, written only.
- The length is a constant because `n_elements` is a constexpr of this configuration.
