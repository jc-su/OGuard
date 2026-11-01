# q18_core_cuda_total_counts_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `total_counts_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `total_counts_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, `<vector>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q18_launcher` (sorted path) and `dense_groupby_launcher` (dense path) each launch
the kernel once with `<<<1, 1, 0, stream>>>`: block (1, 1, 1), no dynamic shared
memory; the same configuration, so it is one corpus kernel.

## Lengths and ranges

- `unique_offsets`, `unique_counts`, `big_offsets`, `big_counts`: the per-block counts and their exclusive scans, `n_blocks` elements each (`line_blocks` on the sorted path, `key_blocks` on the dense path); read only.
- `n_blocks`: size symbol with least value 1.
- `totals`: `torch::empty({2})` of `int64`, 2 elements, written only.
