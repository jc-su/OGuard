# q22_cuda_avg_acctbal_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `avg_acctbal_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `avg_acctbal_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`, the macro `BLOCK_SIZE`, and the device function
  `find_code_idx`; left out the Torch, ATen, c10, and `<vector>` includes and
  the host code.

## Launch

`q22_execute` launches the kernel once with
`<<<(n_customers + 1023) / 1024, BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1),
no dynamic shared memory (the two 1024-entry `__shared__` arrays are static).

## Lengths and ranges

- `c_acctbal`, `cntrycode`: `n_customers` rows each; the launcher sets
  `n_customers = c_acctbal.numel()` and checks `cntrycode.numel()`.
- `n_customers`: size symbol with least value 1; the launcher returns before
  the launches when `n_customers == 0`.
- `codes`: the sorted distinct country codes, `num_codes` int32 entries (7 for
  the default parameters).
- `num_codes`: `codes.numel()` as `int`, size symbol with least value 1 (the
  launcher returns for 0) and, in `launch.json`, the maximum 1024 that the
  launcher checks with `TORCH_CHECK(num_codes <= 1024)`.
- `avg_stats`: `torch::zeros({2})`, which the kernel adds to with `atomicAdd`
  (`readwrite`).
