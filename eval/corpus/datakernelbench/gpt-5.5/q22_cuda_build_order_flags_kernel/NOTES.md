# q22_cuda_build_order_flags_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `build_order_flags_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `build_order_flags_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, and `<vector>`
  includes, the macro `BLOCK_SIZE` and the device function `find_code_idx`
  (not used by the kernel), and the host code.

## Launch

`q22_execute` launches the kernel once, when `n_orders > 0`, with
`<<<(n_orders + 1023) / 1024, BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1), no
dynamic shared memory.

## Lengths and ranges

- `o_custkey`: `n_orders` rows, the `o_custkey` column of `orders`.
- `n_orders`: size symbol with least value 1 (the launch is inside
  `if (n_orders > 0)`).
- `flags`: `torch::zeros({flags_len})` of uint8, which the kernel only stores 1
  into (`write`).
- `flags_len`: `max(1, max(c_custkey) + 1)` from `run_query`; the launcher
  checks `flags_len > 0`, so a size symbol with least value 1.
