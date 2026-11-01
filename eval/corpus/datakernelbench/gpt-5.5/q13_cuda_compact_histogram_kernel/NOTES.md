# q13_cuda_compact_histogram_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q13/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `compact_histogram_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `compact_histogram_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the device function `atomic_inc_i64`; left
  out the Torch and CUDA runtime includes, `<algorithm>`, `<vector>`, the
  macros `BLOCK_SIZE` and `CUDA_CHECK_LAST`, and the host code, which the
  kernel does not use.

## Launch

`q13_launcher` launches the kernel once with
`<<<launch_blocks_for(hist_len), BLOCK_SIZE>>>`: at most 131072 blocks of
(1024, 1, 1) threads, no dynamic shared memory.

## Lengths and ranges

- `hist`: the `hist_len` counts that `histogram_customer_counts_kernel`
  wrote; read only.
- `hist_len`: `max_count + 1`, size symbol with least value 1.
- `unique_counts`, `custdist`: `torch::empty({hist_len})` each, `hist_len`
  elements; the kernel only writes them.
- `n_unique`: `torch::zeros({1})`, one element, the output cursor that the
  kernel increments with `atomicAdd` (`readwrite`). The plan relies on it
  being zero on entry and on each `i < hist_len` being visited once, so that at
  most `hist_len` positions are taken.
