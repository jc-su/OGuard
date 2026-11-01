# q18_cuda_q18_check_sorted_max_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q18_check_sorted_max_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q18_check_sorted_max_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, ATen, c10, `cuda_runtime.h`,
  `<vector>`, and `<algorithm>` includes and the host code, which the kernel
  does not use.

## Launch

`q18_aggregate_launcher` launches the kernel once with
`<<<pick_blocks(n), BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory; `pick_blocks` caps the grid at 131072 blocks, and the kernel
strides over the rows.

## Lengths and ranges

- `keys`: `l_orderkey` as int64, `n = orderkeys.numel()` rows.
- `n`: size symbol with least value 1; the launcher returns for `n == 0`.
- `max_out`: `torch::zeros({1})` of int64, updated with `atomicMax` (`readwrite`).
- `sorted_flag`: `torch::ones({1})` of int32, set with `atomicExch` (`readwrite`).
