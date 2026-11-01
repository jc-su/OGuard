# q03_cuda_q3_revenue_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q03/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q3_revenue_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q3_revenue_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, and
  `cuda_runtime.h` includes, the macro `BLOCK_SIZE` (used only by the host
  code), and the host code.

## Launch

`q3_revenue_launcher` launches the kernel when `n > 0` with
`<<<(n + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `extendedprice_ptr`, `discount_ptr`: `n_elements` rows each, the joined
  `l_extendedprice` and `l_discount`; the launcher sets
  `n = extendedprice.numel()` and checks that `discount` has the same `numel()`.
- `revenue_ptr`: `torch::empty({n})`, written only.
- `n_elements`: size symbol with least value 1 (the launch needs `n > 0`).
