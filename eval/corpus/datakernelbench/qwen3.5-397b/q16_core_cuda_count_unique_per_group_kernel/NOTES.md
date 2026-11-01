# q16_core_cuda_count_unique_per_group_kernel

The kernel `count_unique_per_group_kernel` of the CUDA plan for q16 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q16/round_03.py`,
copied as `plan.py`).

## Extraction edits

- `count_unique_per_group_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 69: `count_unique_per_group_kernel<<<blocks, BLOCK_SIZE>>>` in `q16_launcher`, when `n > 0`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- `group_ids` and `supp_ids` are the joined rows sorted by `(group_id, supp_id)`, `n` rows each (the launcher checks the lengths agree and returns before the launch when `n == 0`).
- `group_ids` holds `_gid = range(len(group_df))` joined onto the rows, in `[0, num_groups)` with `num_groups = len(group_df)`: `index-in` `out_counts`; a non-empty join has at least one group.
- `out_counts` is `torch::zeros({num_groups})`, an `atomicAdd` accumulator (through an `unsigned long long*` cast): `readwrite`.
