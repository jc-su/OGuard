# q06_cuda_q6_revenue_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q06/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q6_revenue_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q6_revenue_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch,
  ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q6_revenue_launcher` launches the kernel once with
`<<<(n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `shipdate_days_ptr`, `discount_ptr`, `quantity_ptr`, `extendedprice_ptr`:
  `n_elements` rows each. The launcher sets `n_elements = shipdate_days.numel()`
  and checks that the other three columns have the same `numel()`.
- `n_elements`: size symbol with least value 1; the launcher returns before the
  launch when `n_elements == 0`.
- `out_ptr`: `torch::zeros({1})`, one element, which the kernel adds to with
  `atomicAdd` (`readwrite`).
- `shipdate_lo_days`, `shipdate_hi_days`, `discount_lo`, `discount_hi`,
  `quantity_hi`: comparison thresholds only; the plan states no range.
