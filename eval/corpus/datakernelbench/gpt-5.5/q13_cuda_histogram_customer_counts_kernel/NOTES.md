# q13_cuda_histogram_customer_counts_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q13/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `histogram_customer_counts_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `histogram_customer_counts_kernel` to `kernel` and declared it
  `extern "C"`.
- Kept `#include <cstdint>` and the device function `atomic_add_i64`; left
  out the Torch and CUDA runtime includes, `<algorithm>`, `<vector>`, the
  macros `BLOCK_SIZE` and `CUDA_CHECK_LAST`, and the host code, which the
  kernel does not use.

## Launch

`q13_launcher` launches the kernel once, when `n_customers > 0`, with
`<<<launch_blocks_for(n_customers), BLOCK_SIZE>>>`: at most 131072 blocks of
(1024, 1, 1) threads, no dynamic shared memory.

## Lengths and ranges

- `c_custkey`: `n_customers` rows of `customer`.
- `n_customers`: size symbol with least value 1; the launch happens only when
  `n_customers > 0`.
- `counts`: the `counts_len` per-customer order counts that
  `count_orders_kernel` wrote; read only.
- `counts_len`: `c_custkey.max() + 1`, least value 1.
- `hist`: `torch::zeros({hist_len})`, which the kernel adds to with
  `atomicAdd` (`readwrite`).
- `hist_len`: `max_count + 1`, where `max_count` is the result of the two
  reduction kernels (0 when there are no customers); size symbol with least
  value 1. The plan's reduction takes the largest count over the customers,
  but the kernel does not rely on it: it checks `0 <= cnt < hist_len`.
