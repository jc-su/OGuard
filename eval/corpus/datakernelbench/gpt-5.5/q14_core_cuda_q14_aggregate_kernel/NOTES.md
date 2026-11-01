# q14_core_cuda_q14_aggregate_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q14/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q14_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q14_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q14_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1),
no dynamic shared memory.

## Lengths and ranges

- `price_ptr`, `discount_ptr` (`float64`), `shipdate_ptr` (`int32`), `promo_ptr` (`bool`): the joined columns, `n_elements = price.numel()` rows each (checked by the launcher).
- `n_elements`: size symbol with least value 1; the launch runs only when `n > 0`.
- `sums_ptr`: `torch::zeros({2})`, 2 elements, which the kernel adds to with `atomicAdd` (`readwrite`).
- `date_lo`, `date_hi`: comparison thresholds only.
