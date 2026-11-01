# q15_core_cuda_q15_max_plain_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q15_max_plain_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_max_plain_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <math_constants.h>` (for `CUDART_INF`), `#include <cstdint>`, and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q15_launcher` launches the kernel in the loop `while (current_n > 1)`, each time
with `<<<next_n, BLOCK_SIZE, 0, stream>>>`, `next_n = (current_n + 255) / 256`: block
(256, 1, 1), no dynamic shared memory. Every launch of the loop has this
configuration, so it is one corpus kernel. At scale factor 10 (100,000 suppliers,
391 block maxima) the loop launches it twice, for 391 and for 2 values.

## Lengths and ranges

- `values`: `current`, the `current_n` block maxima of the previous launch, `n` elements; read only.
- `n`: `current_n`, size symbol with least value 2; the loop launches only while `current_n > 1`.
- `partial`: `torch::empty({next_n})`, one element per block of the plan's grid, `(n + 255) / 256`; the quotient gives the buffer its own size symbol with least value 1. The kernel only writes it.
