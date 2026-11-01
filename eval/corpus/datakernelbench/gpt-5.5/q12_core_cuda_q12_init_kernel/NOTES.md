# q12_core_cuda_q12_init_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q12/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q12_init_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q12_init_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, the device function `atomic_add_i64_nonnegative`, `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q12_launcher` launches the kernel once with `<<<1, 32>>>`: block (32, 1, 1), no
dynamic shared memory.

## Lengths and ranges

- `group_ids`, `high_counts`, `low_counts`: `torch::empty({2})` of `int64` each, 2 elements, which the kernel only writes.
