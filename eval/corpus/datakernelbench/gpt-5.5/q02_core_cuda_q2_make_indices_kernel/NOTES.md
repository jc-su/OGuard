# q02_core_cuda_q2_make_indices_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q02/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q2_make_indices_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q2_make_indices_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out `<algorithm>`, the macro `BLOCK_SIZE`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q2_indices_launcher` launches the kernel once, when `k > 0`, with
`<<<(k + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1),
no dynamic shared memory.

## Lengths and ranges

- `out_ptr`: `torch::empty({k})` of `int64`, `k = min(limit, n)` elements (`n_elements`), which the kernel only writes.
- `n_elements`: `k`, size symbol with least value 1; the launcher launches only when `k > 0`.
