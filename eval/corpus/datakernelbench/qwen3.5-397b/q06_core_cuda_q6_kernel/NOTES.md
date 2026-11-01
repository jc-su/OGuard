# q06_core_cuda_q6_kernel

The kernel `q6_kernel` of the CUDA plan for q06 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q06/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q6_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 79: `q6_kernel<<<blocks, BLOCK_SIZE>>>` in `q6_launcher`, when `n > 0`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory (`sdata` is static).

## Lengths and ranges

- The four columns are lineitem columns of `n = price.numel()` rows; the launcher returns before the launch when `n == 0`.
- `out` is `torch::zeros({1})`, an `atomicAdd` accumulator: `readwrite`.
