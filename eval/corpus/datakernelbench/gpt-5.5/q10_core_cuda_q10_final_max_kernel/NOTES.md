# q10_core_cuda_q10_final_max_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q10/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q10_final_max_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q10_final_max_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Thrust and CUDA runtime includes, `<vector>`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q10_groupby_launcher` launches the kernel once with `<<<1, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `block_max`: the `n_blocks` block maxima of `q10_max_reduce_kernel`; read only.
- `out`: `torch::empty({1})`, one element, written by thread 0.
- `n_blocks`: `(n + 1023) / 1024` with `n >= 1`, size symbol with least value 1.
