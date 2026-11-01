# q13_triton__reduce_partial_hist_kernel

Source: `azure__gpt-5.5/triton/full/queries/q13/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_reduce_partial_hist_kernel` of
`_KERNEL_SOURCE` (plan lines 67 to 82).

## Extraction edits

- Renamed `_reduce_partial_hist_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan lines 215 to 222) launches the kernel once with the 2-D
grid `(small_bins, triton.cdiv(num_customer_blocks, reduce_block_size))`,
(256, 1) at scale factor 10, `num_blocks=num_customer_blocks` (a
`tl.constexpr`, 1,465 at scale factor 10: `triton.cdiv(1,500,000, 1024)`),
`SMALL_BINS=small_bins` (256), `REDUCE_BLOCK=reduce_block_size` (2048), and no
`num_warps` (4).

## Lengths and ranges

- `partial_hist_ptr`: the int64 partial histograms of `_hist_partial_kernel`,
  1,465 * 256 = 375,040 elements; read only.
- `hist_ptr`: `hist_small = torch.zeros(small_bins)` int64, 256 elements,
  added to at `program_id(0)` with `tl.atomic_add` (`readwrite`).
