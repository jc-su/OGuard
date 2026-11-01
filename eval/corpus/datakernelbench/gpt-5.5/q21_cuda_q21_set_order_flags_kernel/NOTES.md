# q21_cuda_q21_set_order_flags_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q21/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q21_set_order_flags_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q21_set_order_flags_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`; left out the Torch, ATen, and c10 includes, the
  macro `BLOCK_SIZE` (not used by the kernel), and the host code.

## Launch

`q21_grouped_launcher` launches the kernel once, when `n_orders > 0`, with
`<<<(n_orders + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no
dynamic shared memory.

## Lengths and ranges

- `order_keys`: `n` rows, the `o_orderkey` of the orders with status 'F'.
- `n`: `order_f_keys.numel()`, size symbol with least value 1 (the launch is
  inside `if (n_orders > 0)`, and `run_query` returns earlier when there are none).
- `max_orderkey`: `int(df_orders["o_orderkey"].max())`, size symbol with least
  value 0 (orders is not empty and keys are positive; 0 is the bound the plan's
  allocation needs).
- `flags`: `torch::zeros({max_orderkey + 1})` of uint8, which the kernel only
  stores 1 into (`write`).
