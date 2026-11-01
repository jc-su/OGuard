# q15_cuda_q15_reduce_final_max_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q15/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q15_reduce_final_max_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_reduce_final_max_kernel` to `kernel` and declared it
  `extern "C"`.
- Kept `#include <stdint.h>`, `#include <cfloat>` (for `DBL_MAX`), and the
  macro `BLOCK_SIZE`; left out the Torch, ATen, c10, and CUDA runtime includes, `<vector>`, the host code, and the other macros and device functions, which
  the kernel does not use.

## Launch

`q15_launcher` launches the kernel once with
`<<<1, BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `block_max`: the `reduce_blocks` maxima of `q15_reduce_max_kernel`, read
  only, `n_blocks` entries.
- `n_blocks`: `reduce_blocks`, which the plan clamps to [1, 4096]; size symbol
  with least value 1, and `launch.json` states its maximum 4096.
- `max_out`: `torch::empty({1})`, one element, which the kernel only writes.
