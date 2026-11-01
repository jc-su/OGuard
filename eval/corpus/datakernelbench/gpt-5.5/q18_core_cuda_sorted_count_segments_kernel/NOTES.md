# q18_core_cuda_sorted_count_segments_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `sorted_count_segments_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `sorted_count_segments_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q18_launcher` launches the kernel once, when `n > 0`, with
`<<<line_blocks, BLOCK_SIZE, 0, stream>>>`, `line_blocks = (n + 1023) / 1024`: block
(1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `keys` (`int64`), `qty` (`float64`): `l_orderkey` and `l_quantity`, `n` rows each (checked equal by the launcher).
- `unique_counts`, `big_counts`: `torch::empty({line_blocks})` each, one element per block of the plan's grid, `(n + 1023) / 1024`; the quotient gives the two buffers one size symbol (`unique_counts`) with least value 1. The kernel only writes them.
- `violation`: `torch::zeros({1})` of `int32`, one element, set with `atomicExch` (`readwrite`).
- `n`: size symbol with least value 1; the launcher returns before the launch when `n == 0`. `threshold`: a comparison threshold only.
