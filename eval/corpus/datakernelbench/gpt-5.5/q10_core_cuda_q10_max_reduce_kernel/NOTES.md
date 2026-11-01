# q10_core_cuda_q10_max_reduce_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q10/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q10_max_reduce_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q10_max_reduce_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Thrust and CUDA runtime includes, `<vector>`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q10_groupby_launcher` launches the kernel once, when `n > 0`, with
`<<<n_blocks, BLOCK_SIZE, 0, stream>>>`, `n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`
(checked to fit an `int`): block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `keys`: `c_custkey`, `n` rows; read only.
- `block_max`: `torch::empty({n_blocks})`, one element per block of the plan's grid, `(n + 1023) / 1024`; the quotient gives the buffer its own size symbol with least value 1. The kernel only writes it.
- `n`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
