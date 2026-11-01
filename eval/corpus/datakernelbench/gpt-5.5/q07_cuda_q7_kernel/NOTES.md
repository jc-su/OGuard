# q07_cuda_q7_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q07/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q7_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q7_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`, the macros `BLOCK_SIZE` and `NUM_GROUPS`, and the
  device function `civil_year_from_days`; left out the Torch, ATen, c10, and
  `cuda_runtime.h` includes and the host code, which the kernel does not use.

## Launch

`q7_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `supp_code`, `cust_code`, `shipdate_days`, `price`, `discount`: `n` rows each,
  the columns of the joined lineitem rows. The launcher sets `n = price.numel()`
  and checks that the other four have the same `numel()`.
- `n`: size symbol with least value 1; the launch is inside `if (n > 0)`.
- `out`: `torch::zeros({NUM_GROUPS})`, 8 elements, which the kernel adds to with
  `atomicAdd` (`readwrite`).
- `base_year`: `_query_core` passes the constant 1995, so `[1995, 1995]`.
- `date_lo`, `date_hi`: epoch days of the ship-date bounds; comparisons only.
- `supp_code` and `cust_code` hold 0 or 1 by construction (`astype("int32")` of
  a comparison); the kernel checks the pair itself, so no range is stated.
