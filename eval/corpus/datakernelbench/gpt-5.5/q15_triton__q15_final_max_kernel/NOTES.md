# q15_triton__q15_final_max_kernel

Source: `azure__gpt-5.5/triton/full/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q15_final_max_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q15_final_max_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once with grid `(1,)`, `n_blocks` as a
`tl.constexpr`, `BLOCK_SIZE=_next_power_of_2(n_blocks)`, and
`num_warps=1 if BLOCK_SIZE <= 64 else 4`. At scale factor 10
`n_blocks = triton.cdiv(100000, 1024) = 98`, so `BLOCK_SIZE = 128` and
`num_warps = 4`.

## Lengths and ranges

- `block_max_ptr`: `torch.empty(n_blocks)`, 98 `float64`, read.
- `max_ptr`: `torch.empty(1)`, one `float64`, written.
- The lengths are constants because `n_blocks` is a constexpr of this
  configuration.
