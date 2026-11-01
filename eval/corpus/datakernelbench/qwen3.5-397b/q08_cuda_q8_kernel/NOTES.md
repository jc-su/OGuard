# q08_cuda_q8_kernel

The kernel `q8_kernel` of the CUDA plan for q08 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q08/round_03.py`,
copied as `plan.py`).

## Extraction edits

- `q8_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 115: `q8_kernel<<<blocks, BLOCK_SIZE, smem>>>` in `q8_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and `smem = 4 * BLOCK_SIZE * sizeof(double)` = 32768 bytes of dynamic shared memory.

## Lengths and ranges

- The four columns are columns of the joined table `li`, of `n = years.numel()` rows.
- `out_total` and `out_brazil` are `torch::zeros({2})`, `atomicAdd` accumulators: `readwrite`.
