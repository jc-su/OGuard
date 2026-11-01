# q13_core_cuda_q13_count_orders_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q13/round_03.py` (`output_round` 3),
copied as `plan.py`. The kernel is `q13_count_orders_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q13_count_orders_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macros `BLOCK_SIZE` and `HIST_SHARED_CAP`, `<algorithm>`, `<utility>`, `<vector>`, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q13_launcher` launches the kernel once, when `n_orders > 0`, with
`<<<(n_orders + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `o_custkey`: the customer keys of the orders without special requests, `n_orders` rows; a foreign key into `customer`, so `read` (the kernel bounds each key itself).
- `counts`: `torch::zeros({num_customers + 1})` of `int32`, `num_customers + 1` elements, which the kernel adds to with `atomicAdd` (`readwrite`).
- `num_customers`: `int(df_customer["c_custkey"].max())`, checked `>= 0` by the launcher: size symbol with least value 0.
- `n_orders`: size symbol with least value 1; the launch runs only when `n_orders > 0`.
