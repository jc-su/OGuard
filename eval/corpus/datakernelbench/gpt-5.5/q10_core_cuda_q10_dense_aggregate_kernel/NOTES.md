# q10_core_cuda_q10_dense_aggregate_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q10/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q10_dense_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q10_dense_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, the Thrust and CUDA runtime includes, `<vector>`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q10_groupby_launcher` launches the kernel once, when `n > 0` and the maximum key is
non-negative, with `<<<n_blocks, BLOCK_SIZE, 0, stream>>>`,
`n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`: block (1024, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `custkey` (`int64`), `extendedprice`, `discount` (`float64`): the joined columns, `n` rows each (checked by the launcher).
- `dense_size`: `max(custkey) + 1` from the two reduction kernels, checked `> 0`: size symbol with least value 1. The plan sizes `revenue_dense` from the maximum of the same keys, and TPC-H keys are positive, so `custkey` is `index-in` `revenue_dense`.
- `revenue_dense`: `torch::zeros({dense_size})`, which the kernel adds to with `atomicAdd` (`readwrite`); `seen`: `torch::zeros({dense_size})` of `uint8`, written only.
- `n`: size symbol with least value 1.
