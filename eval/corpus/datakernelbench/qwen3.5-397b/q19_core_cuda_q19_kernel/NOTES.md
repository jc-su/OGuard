# q19_core_cuda_q19_kernel

The kernel `q19_kernel` of the CUDA plan for q19 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q19/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q19_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 117: `q19_kernel<<<blocks, BLOCK_SIZE>>>` in `q19_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory (`block_revenue` is static).

## Lengths and ranges

- The six columns are columns of the joined lineitem and part table, of `n = l_quantity.numel()` rows.
- `out` is `torch::zeros({1})`, an `atomicAdd` accumulator: `readwrite`.
