# q04_core_triton__zero_u8_kernel

Source: `azure__gpt-5.5/triton/core/queries/q04/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_zero_u8_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_zero_u8_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q4_triton` launches the kernel once with grid
`(triton.cdiv(flag_n, 1024),)`, `BLOCK_SIZE=1024`, and no `num_warps` (4).

## Lengths and ranges

- `ptr`: `torch.empty(flag_n, dtype=torch.uint8)`, `n_elements = flag_n` bytes, written only.
- `n_elements`: `flag_n = max(max(l_orderkey) + 1, 1)`, size symbol with least value 1 (60,000,001 at scale factor 10, so `i32`).
