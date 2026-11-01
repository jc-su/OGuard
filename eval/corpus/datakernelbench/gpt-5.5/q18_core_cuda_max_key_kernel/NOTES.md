# q18_core_cuda_max_key_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `max_key_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `max_key_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`dense_groupby_launcher` launches the kernel once with
`<<<line_blocks, BLOCK_SIZE, 0, stream>>>`, `line_blocks = (n + 1023) / 1024`: block
(1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `keys`: `l_orderkey`, `n` rows; read only.
- `max_out`: `torch::zeros({1})` of `int64` reinterpreted as `unsigned long long`, one element, updated with `atomicMax` (`readwrite`).
- `n`: size symbol with least value 1.

## Branch

The plan takes the dense path only when `sorted_count_segments_kernel` found a
decrease in `l_orderkey`, which the benchmark's lineitem table at scale factor 10
probably does not have.
