# q16_core_cuda_q16_distinct_count_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q16/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q16_distinct_count_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q16_distinct_count_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`, the constant `EMPTY_KEY`, and the device function `mix64`; left out the macro `BLOCK_SIZE`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q16_distinct_count_launcher` launches the kernel once, when `n > 0` and
`num_groups > 0`, with `<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `group_ids`, `supp_ids`: the joined `_gid` and `ps_suppkey` columns, `n_elements = group_ids.numel()` rows each (checked equal by the launcher).
- `group_ids`: `_gid = range(len(group_df))` by construction and `num_groups = len(group_df)`, so every value lies in `[0, num_groups)`: `index-in` `counts`.
- `table`: `torch::empty({table_capacity})` of `int64` filled with `0xFF` bytes (`EMPTY_KEY`) by `cudaMemsetAsync`, `table_capacity` entries, probed with `atomicCAS` (`readwrite`).
- `table_capacity`: the smallest power of two of at least `max(16, 4 * n)`, a size symbol with least value 16.
- `counts`: `torch::empty({num_groups})` zeroed by `cudaMemsetAsync`, `num_groups` elements, added to with `atomicAdd` (`readwrite`).
- `n_elements`, `num_groups`: size symbols with least value 1; the launcher returns before the launch when either is 0.
