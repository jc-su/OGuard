# q14_core_cuda_q14_finalize_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q14/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q14_finalize_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q14_finalize_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q14_launcher` launches the kernel once with `<<<1, 1, 0, stream>>>`: block
(1, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `sums_ptr`: the 2 sums of `q14_aggregate_kernel` (zeros when `n == 0`); read only.
- `out_ptr`: `torch::empty({1})`, one element, written only.
