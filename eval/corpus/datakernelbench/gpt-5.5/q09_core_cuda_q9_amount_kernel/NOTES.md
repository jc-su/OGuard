# q09_core_cuda_q9_amount_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q09/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q9_amount_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q9_amount_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out `<limits>`, the macro `BLOCK_SIZE`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q9_amount_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1),
no dynamic shared memory.

## Lengths and ranges

- `l_extendedprice`, `l_discount`, `ps_supplycost`, `l_quantity`: `n_elements` rows each; the launcher sets `n = l_extendedprice.numel()` and checks the other three columns against it.
- `out`: `torch::empty({n})`, `n_elements` elements, which the kernel only writes.
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
