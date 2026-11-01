# q08_core_triton__q8_finalize_2y_kernel

Source: `azure__gpt-5.5/triton/core/queries/q08/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `_q8_finalize_2y_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q8_finalize_2y_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, when `n > 0`, with grid `(1,)`, no
`tl.constexpr` arguments, and no `num_warps` (4).

## Lengths and ranges

- `agg_ptr`: the 4 sums of `_q8_reduce_1995_1996_kernel`; read only.
- `out_ptr`: `torch.empty(2, dtype=torch.float64)`, 2 elements, written only.
