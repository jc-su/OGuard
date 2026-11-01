# q18_cuda_q18_atomic_aggregate_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q18_atomic_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q18_atomic_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, `cuda_runtime.h`,
  `<vector>`, and `<algorithm>` includes, the macro `BLOCK_SIZE`, and the host code, which the kernel
  does not use.

## Launch

`q18_aggregate_launcher` launches the kernel once, on the branch where
`q18_check_sorted_max_kernel` found the keys not sorted, with
`<<<pick_blocks(n), BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory; the kernel strides over the rows.

## Lengths and ranges

- `keys`, `qty`: `n = orderkeys.numel()` rows. `sums_n = dense_n = max_key + 1`,
  where `max_key` is the maximum of the same keys (and of 0) that
  `q18_check_sorted_max_kernel` computed, checked `>= 0`; order keys are positive,
  so every key lies in `[0, sums_n - 1]`: `index-in` into `sums`.
- `sums`: `torch::zeros({dense_n})`, added to with `atomicAdd` (`readwrite`).
- `n`, `sums_n`: size symbols with least value 1 (the launcher returns for
  `n == 0`; `max_key >= 0`).
