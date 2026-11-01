# q15_core_cuda_q15_max_present_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q15_max_present_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_max_present_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <math_constants.h>` (for `CUDART_INF`), `#include <cstdint>`, and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q15_launcher` launches the kernel once with
`<<<max_blocks, BLOCK_SIZE, 0, stream>>>`, `max_blocks = max(1, (n_sup + 255) / 256)`:
block (256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `totals`, `present`: the `n_sup` totals and flags of `q15_aggregate_kernel` (zeros when that kernel did not run), `n` elements each; read only.
- `n`: `n_sup`, size symbol with least value 0; the plan launches this kernel for any supplier count.
- `partial`: `torch::empty({max_blocks})`, one element per block of the plan's grid, `(n + 255) / 256` (1 when `n` is 0); the quotient gives the buffer its own size symbol with least value 1. The kernel only writes it.
