# q08_core_cuda_q8_aggregate_kernel

The kernel `q8_aggregate_kernel` of the CUDA plan for q08 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q08/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q8_aggregate_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 65: `q8_aggregate_kernel<<<blocks, BLOCK_SIZE>>>` in `q8_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The three columns have `n = year_idx.numel()` rows of the joined table; `year_idx` is the inverse of `torch.unique(o_year)` cast to int32, so its values lie in `[0, num_groups)` with `num_groups` the number of distinct years: `year_idx_ptr` is `index-in` `sum_total_ptr` (`sum_brazil_ptr` has the same length); the kernel also checks the range itself.
- `sum_brazil` and `sum_total` are `torch::zeros({num_groups})`, `atomicAdd` accumulators: `readwrite`.
