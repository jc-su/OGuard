# q18_cuda_q18_fill_big_sorted_runs_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q18_fill_big_sorted_runs_kernel` of
`_CUDA_SOURCE`.

## Extraction edits

- Renamed `q18_fill_big_sorted_runs_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, `cuda_runtime.h`,
  `<vector>`, and `<algorithm>` includes, the macro `BLOCK_SIZE`, and the host code, which the kernel
  does not use.

## Launch

`q18_aggregate_launcher` launches the kernel once, on the sorted branch after
`q18_count_big_sorted_runs_kernel` and only when its count `out_n > 0`, with
`<<<pick_blocks(n), BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory; the kernel strides over the rows.

## Lengths and ranges

- `keys`, `qty`: `n = orderkeys.numel()` rows.
- `n`: size symbol with least value 1; the launcher returns for `n == 0`.
- `threshold`: the HAVING threshold, a float with no range.
- `out_keys`, `out_sums`: `torch::empty({out_n})`, where `out_n` is the count that
  `q18_count_big_sorted_runs_kernel` wrote; the kernel only stores into them
  (`write`). `out_n` is a size symbol with least value 1 (the launch runs only
  for `out_n > 0`), which the kernel does not receive.
- `pos_out`: `torch::zeros({1})` of int64 passed as `unsigned long long*`, the
  append position, taken with `atomicAdd` (`readwrite`).
