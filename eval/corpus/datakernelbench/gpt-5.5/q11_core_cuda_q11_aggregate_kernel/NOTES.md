# q11_core_cuda_q11_aggregate_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q11/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q11_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q11_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<cuda.h>`, the CUDA runtime include, `<vector>`, the Thrust includes, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q11_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1),
no dynamic shared memory.

## Lengths and ranges

- `partkey_ptr` (`int64`), `supplycost_ptr`, `availqty_ptr` (`float64`): the joined partsupp columns, `n_elements = ps_partkey.numel()` rows each (checked by the launcher).
- `dense_len`: `max_key + 1`, where `max_key` is the Thrust maximum of the same `ps_partkey` column, checked `>= 0`: a size symbol that the kernel does not receive, with least value 1. TPC-H part keys are positive, so `partkey_ptr` is `index-in` `dense_values_ptr`.
- `dense_values_ptr`: `torch::zeros({dense_len})`, which the kernel adds to with `atomicAdd` (`readwrite`); `present_ptr`: `torch::zeros({dense_len})` of `int32`, written only.
- `total_ptr`: `torch::zeros({1})`, one element, added to with `atomicAdd` (`readwrite`).
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
