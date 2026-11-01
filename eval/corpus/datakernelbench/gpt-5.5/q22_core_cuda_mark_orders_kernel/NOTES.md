# q22_core_cuda_mark_orders_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `mark_orders_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `mark_orders_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out `<climits>`, the macros `BLOCK_SIZE` and `MAX_CODE`, `<algorithm>`, `<vector>`, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q22_launcher` launches the kernel once, when `n_orders > 0`, with
`<<<(n_orders + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `o_custkey`: the distinct `o_custkey` values of orders, `n_orders` rows; a foreign key into `customer`, so `read` (the kernel bounds each key itself).
- `has_order`: `torch::zeros({key_space})` of `uint8`, `key_space` bytes, written only.
- `key_space`: `max_key + 1` with `max_key = max(0, max(c_custkey))`: size symbol with least value 1. `n_orders`: least value 1.
