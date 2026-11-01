# q07_core_cuda_q7_scan_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q07/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q7_scan_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q7_scan_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`, the macros `BLOCK_SIZE` and `NUM_GROUPS`, and the device function `year_from_epoch_days`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q7_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1),
no dynamic shared memory.

## Lengths and ranges

- `supp_ptr`, `cust_ptr`, `shipdate_ptr` (`int32`), `price_ptr`, `discount_ptr` (`float64`): the joined columns, `n_elements = l_extendedprice.numel()` rows each (checked by the launcher).
- `n_elements`: size symbol with least value 1; the launch runs only when `n > 0`.
- `revenue8_ptr`: `torch::zeros({NUM_GROUPS})`, 8 elements, which the kernel adds to with `atomicAdd` (`readwrite`).
- `date_lo`, `date_hi`: comparison thresholds only.
