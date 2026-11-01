# q13_core_cuda_q13_hist_customer_counts_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q13/round_03.py` (`output_round` 3),
copied as `plan.py`. The kernel is `q13_hist_customer_counts_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q13_hist_customer_counts_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `HIST_SHARED_CAP`; left out `BLOCK_SIZE`, `<algorithm>`, `<utility>`, `<vector>`, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q13_launcher` launches the kernel once, when `n_customers > 0`, with
`<<<(n_customers + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `c_custkey`: the customer keys, `n_customers = c_custkey.numel()` rows. The plan sizes `counts` from the maximum of the same keys (`num_customers + 1`), and TPC-H keys are positive, so `c_custkey` is `index-in` `counts`.
- `counts`: the `num_customers + 1` order counts of `q13_count_orders_kernel`; read only here.
- `hist`: `torch::zeros({hist_size})` of `int64`, `hist_size = max(counts) + 1` elements, which the kernel adds to with `atomicAdd` (`readwrite`).
- `n_customers`: size symbol with least value 1 (the launcher returns for `n_customers <= 0`); `num_customers`: checked `>= 0`, least value 0; `hist_size`: at least 1.
