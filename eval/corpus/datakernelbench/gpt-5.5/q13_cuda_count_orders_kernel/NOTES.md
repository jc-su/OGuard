# q13_cuda_count_orders_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q13/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `count_orders_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `count_orders_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the device function `atomic_add_i64`; left
  out the Torch and CUDA runtime includes, `<algorithm>`, `<vector>`, the
  macros `BLOCK_SIZE` and `CUDA_CHECK_LAST`, and the host code, which the
  kernel does not use.

## Launch

`q13_launcher` launches the kernel once, when `n_orders > 0`, with
`<<<launch_blocks_for(n_orders), BLOCK_SIZE>>>`: at most 131072 blocks of
(1024, 1, 1) threads, no dynamic shared memory.

## Lengths and ranges

- `o_custkey`: `n_orders` rows, `o_custkey` of the orders whose comment does
  not match the pattern; `n_orders = o_custkey.numel()`.
- `n_orders`: size symbol with least value 1; the launch happens only when
  `n_orders > 0`.
- `counts`: `torch::zeros({counts_len})`, `counts_len` elements, which the
  kernel adds to with `atomicAdd` (`readwrite`).
- `counts_len`: `num_customers_max_key + 1`, where `num_customers_max_key` is
  `int(df_customer["c_custkey"].max())` and the launcher checks it is
  non-negative; size symbol with least value 1. The plan does not bound
  `o_custkey` by it; the kernel checks `0 <= key < counts_len`.
