# q09_cuda_q9_kernel

The kernel `q9_kernel` of the CUDA plan for q09 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q09/round_05.py`,
copied as `plan.py`).

## Extraction edits

- `q9_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 101: `q9_kernel<<<blocks, BLOCK_SIZE>>>` in `q9_launcher`, with `blocks = (n + 1023) / 1024` capped at 132 * 8, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The six columns have `n = price.numel()` rows; the launcher checks that the other five have the same `numel()`.
- `group_sums` is `torch::zeros({n_groups})`, an `atomicAdd` accumulator: `readwrite`; `n_groups` is passed as the kernel's `n_groups`.
