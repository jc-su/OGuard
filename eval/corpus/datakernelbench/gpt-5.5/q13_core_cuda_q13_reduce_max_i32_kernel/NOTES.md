# q13_core_cuda_q13_reduce_max_i32_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q13/round_03.py` (`output_round` 3),
copied as `plan.py`. The kernel is `q13_reduce_max_i32_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q13_reduce_max_i32_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `HIST_SHARED_CAP`, `<algorithm>`, `<utility>`, `<vector>`, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q13_device_max_i32` launches the kernel in the loop `while (current_n > 1)`, each
time with `<<<blocks64, BLOCK_SIZE, 0, stream>>>`, `blocks64 = (current_n + 255) / 256`:
block (256, 1, 1), no dynamic shared memory. Every launch of the loop has this
configuration, so it is one corpus kernel. At scale factor 10 the loop launches it
three times, for 1,500,001, 5,860, and 23 values.

## Lengths and ranges

- `inp`: `counts` at the first launch (`num_customers + 1` entries) and the previous launch's `next` afterwards, `n` elements; read only.
- `n`: `current_n`, size symbol with least value 2; the loop launches only while `current_n > 1`.
- `out`: `torch::empty({blocks64})`, one element per block of the plan's grid, `(n + 255) / 256`; the quotient gives the buffer its own size symbol with least value 1. The kernel only writes it.
