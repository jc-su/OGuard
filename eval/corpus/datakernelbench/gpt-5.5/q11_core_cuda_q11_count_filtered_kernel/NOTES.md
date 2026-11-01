# q11_core_cuda_q11_count_filtered_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q11/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q11_count_filtered_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q11_count_filtered_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<cuda.h>`, the CUDA runtime include, `<vector>`, the Thrust includes, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q11_launcher` launches the kernel once with
`<<<(dense_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `dense_values_ptr`, `present_ptr`: the `dense_len` sums and flags of `q11_aggregate_kernel`; read only.
- `dense_len`: `max(ps_partkey) + 1`, size symbol with least value 1.
- `count_ptr`: `torch::zeros({1})` of `int64` reinterpreted as `unsigned long long`, one element, added to with `atomicAdd` (`readwrite`).
- `threshold`: a comparison threshold only.
