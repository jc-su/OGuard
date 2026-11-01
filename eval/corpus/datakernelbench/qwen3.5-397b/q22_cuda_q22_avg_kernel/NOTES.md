# q22_cuda_q22_avg_kernel

The kernel `q22_avg_kernel` of the CUDA plan for q22 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q22/round_03.py`,
copied as `plan.py`).

## Extraction edits

- `q22_avg_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>`, `#include <algorithm>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 186: `q22_avg_kernel<<<blocks, BLOCK_SIZE>>>` in `q22_avg_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- `c_acctbal` and `in_mask` have `n = c_acctbal.numel()` rows.
- `sum_tensor` and `count_tensor` are zeroed one-element tensors, `atomicAdd` accumulators: `readwrite`.
