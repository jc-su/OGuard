# q22_core_cuda_max_i64_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `max_i64_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `max_i64_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <climits>` (for `LLONG_MIN`), `#include <cstdint>`, and the macro `BLOCK_SIZE`; left out `MAX_CODE`, `<algorithm>`, `<vector>`, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q22_launcher` launches the kernel once, when `n_customers > 0`, with
`<<<cust_blocks, BLOCK_SIZE, 0, stream>>>`, `cust_blocks = (n_customers + 255) / 256`:
block (256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `data`: `c_custkey`, `n = n_customers` rows; read only.
- `block_max`: `torch::empty({cust_blocks})`, one element per block of the plan's grid, `(n + 255) / 256`; the quotient gives the buffer its own size symbol with least value 1. Written only.
- `n`: size symbol with least value 1; the launcher returns before the launch when `n_customers == 0`.
