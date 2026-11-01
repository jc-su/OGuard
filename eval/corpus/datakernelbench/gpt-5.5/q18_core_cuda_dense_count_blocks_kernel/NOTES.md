# q18_core_cuda_dense_count_blocks_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `dense_count_blocks_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `dense_count_blocks_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`dense_groupby_launcher` launches the kernel once with
`<<<key_blocks, BLOCK_SIZE, 0, stream>>>`, `key_blocks = (key_range + 1023) / 1024`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `flags` (`int32`), `sums` (`float64`): the `key_range` entries of `dense_accumulate_kernel`; read only.
- `unique_counts`, `big_counts`: `torch::empty({key_blocks})` each, one element per block of the plan's grid, `(key_range + 1023) / 1024`; the quotient gives the two buffers one size symbol (`unique_counts`) with least value 1. Written only.
- `key_range`: size symbol with least value 1. `threshold`: a comparison threshold only.

## Branch

The plan takes the dense path only when `sorted_count_segments_kernel` found a
decrease in `l_orderkey`, which the benchmark's lineitem table at scale factor 10
probably does not have.
