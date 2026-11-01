# q12_cuda_q12_count_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q12/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q12_count_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q12_count_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`, the macro `BLOCK_SIZE`, and the device function
  `atomic_add_i64`; left out the Torch, ATen, and c10 includes and the host
  code, which the kernel does not use.

## Launch

`q12_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `shipmode_code`, `priority_high`, `commit_days`, `receipt_days`,
  `ship_days`: `n_elements` rows each, the columns of the join of the filtered
  `lineitem` with `orders`. The launcher sets `n = shipmode_code.numel()` and
  checks that the other four columns have the same `numel()`.
- `n_elements`: size symbol with least value 1; the launcher launches only when
  `n > 0`.
- `out_counts`: `torch::zeros({4})` of `int64`, 4 elements, which the kernel
  adds to with `atomicAdd` (`readwrite`).
- `receipt_lo_days`, `receipt_hi_days`: comparison thresholds only; the plan
  states no range.
