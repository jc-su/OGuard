# q15_triton__q15_block_max_kernel

Source: `azure__gpt-5.5/triton/full/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q15_block_max_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q15_block_max_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once with grid `(n_blocks,)`,
`n_blocks = triton.cdiv(n_sup, 1024)`, `n_sup` as a `tl.constexpr`,
`BLOCK_SIZE=1024`, and `num_warps=4`. `n_sup` is the rows of `supplier`,
100,000 at scale factor 10, so `n_blocks = 98`.

## Lengths and ranges

- `totals_ptr` (`float64`), `present_ptr` (`uint8`): 100000 elements each,
  read.
- `block_max_ptr`: `torch.empty(n_blocks)`, 98 `float64`, which the kernel
  writes at `pid`.
- The lengths are constants because `n_sup` is a constexpr of this
  configuration.
