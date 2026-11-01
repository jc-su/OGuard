# q18_cuda_q18_count_big_sorted_runs_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q18_count_big_sorted_runs_kernel` of
`_CUDA_SOURCE`.

## Extraction edits

- Renamed `q18_count_big_sorted_runs_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, ATen, c10, `cuda_runtime.h`,
  `<vector>`, and `<algorithm>` includes and the host code, which the kernel
  does not use.

## Launch

`q18_aggregate_launcher` launches the kernel once, when `q18_check_sorted_max_kernel`
found the keys nondecreasing, with `<<<pick_blocks(n), BLOCK_SIZE, 0, stream>>>`:
block (256, 1, 1), no dynamic shared memory; the kernel strides over the rows.

## Lengths and ranges

- `keys`, `qty`: `l_orderkey` and `l_quantity`, `n = orderkeys.numel()` rows; the
  launcher checks both have the same `numel()`.
- `n`: size symbol with least value 1; the launcher returns for `n == 0`.
- `threshold`: the HAVING threshold (300.0), a float with no range.
- `count_out`: `torch::zeros({1})` of int64 passed as `unsigned long long*`,
  added to with `atomicAdd` (`readwrite`).
