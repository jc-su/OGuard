# q18_cuda_q18_count_big_dense_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q18_count_big_dense_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q18_count_big_dense_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, ATen, c10, `cuda_runtime.h`,
  `<vector>`, and `<algorithm>` includes and the host code, which the kernel
  does not use.

## Launch

`q18_aggregate_launcher` launches the kernel once, on the unsorted branch after
`q18_atomic_aggregate_kernel`, with `<<<pick_blocks(dense_n), BLOCK_SIZE, 0, stream>>>`:
block (256, 1, 1), no dynamic shared memory; the kernel strides over the elements.

## Lengths and ranges

- `sums`: `torch::zeros({dense_n})`, filled by `q18_atomic_aggregate_kernel`;
  the kernel receives `n = dense_n`.
- `n`: size symbol with least value 1 (`dense_n = max_key + 1` with `max_key >= 0`).
- `threshold`: the HAVING threshold, a float with no range.
- `count_out`: `torch::zeros({1})` of int64 passed as `unsigned long long*`,
  added to with `atomicAdd` (`readwrite`).
