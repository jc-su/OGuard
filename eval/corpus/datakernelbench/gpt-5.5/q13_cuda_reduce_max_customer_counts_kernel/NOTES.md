# q13_cuda_reduce_max_customer_counts_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q13/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `reduce_max_customer_counts_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `reduce_max_customer_counts_kernel` to `kernel` and declared it
  `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch
  and CUDA runtime includes, `<algorithm>`, `<vector>`, the macro
  `CUDA_CHECK_LAST`, the device functions, and the host code, which the
  kernel does not use.

## Launch

`q13_launcher` launches the kernel once, when `n_customers > 0`, with
`<<<first_blocks, BLOCK_SIZE>>>`, `first_blocks = launch_blocks_for(n_customers)
= min((n_customers + 1023) / 1024, 131072)`: block (1024, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `c_custkey`: `n_customers` rows of `customer`; `n_customers = c_custkey.numel()`.
- `n_customers`: size symbol with least value 1; the launch happens only when
  `n_customers > 0`, and `run_query` returns before for an empty `customer`.
- `counts`: the `counts_len` per-customer order counts that
  `count_orders_kernel` wrote; read only.
- `counts_len`: `c_custkey.max() + 1`, least value 1 (see
  `q13_cuda_count_orders_kernel`).
- `block_max`: `torch::empty({first_blocks})`, one element per block of the
  plan's grid, `min((n_customers + 1023) / 1024, 131072)`. The signature
  states the quotient `(n_customers + 1023) / 1024`, which gives the buffer its
  own size symbol; its least value is 1. The kernel only writes it.
