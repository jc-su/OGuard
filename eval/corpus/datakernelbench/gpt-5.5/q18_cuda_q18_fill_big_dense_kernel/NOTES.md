# q18_cuda_q18_fill_big_dense_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q18_fill_big_dense_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q18_fill_big_dense_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, `cuda_runtime.h`,
  `<vector>`, and `<algorithm>` includes, the macro `BLOCK_SIZE`, and the host code, which the kernel
  does not use.

## Launch

`q18_aggregate_launcher` launches the kernel once, on the unsorted branch after
`q18_count_big_dense_kernel` and only when its count `out_n > 0`, with
`<<<pick_blocks(dense_n), BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory; the kernel strides over the elements.

## Lengths and ranges

- `sums`: `dense_n` elements; the kernel receives `n = dense_n`.
- `n`: size symbol with least value 1 (`dense_n = max_key + 1` with `max_key >= 0`).
- `threshold`: the HAVING threshold, a float with no range.
- `out_keys`, `out_sums`: `torch::empty({out_n})`, where `out_n` is the count that
  `q18_count_big_dense_kernel` wrote over the same `sums` and `threshold`; the
  kernel only stores into them (`write`). `out_n` is a size symbol with least
  value 1 (the launch runs only for `out_n > 0`), which the kernel does not receive.
  The stored `out_keys` are order keys that `run_query` merges on, not indices
  into a buffer of the kernel, so `write`.
- `pos_out`: `torch::zeros({1})` of int64 passed as `unsigned long long*`, the
  append position, taken with `atomicAdd` (`readwrite`).
