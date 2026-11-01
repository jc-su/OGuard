# q22_core_cuda_q22_phase1_kernel

The kernel `q22_phase1_kernel` of the CUDA plan for q22 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q22/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q22_phase1_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` and `#include <algorithm>` (host binding code and headers the kernel does not use), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 146: `q22_phase1_kernel<<<blocks, BLOCK_SIZE>>>` in `q22_launcher`, with `blocks = (n_customers + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory (`block_sum` and `block_count` are static).

## Lengths and ranges

- The two columns are customer columns of `n = c_custkey.numel()` rows; `in_cntrycode_mask` is a bool tensor.
- `sum_out` and `count_out` are `torch::zeros({1})`, `atomicAdd` accumulators (`count_out` through an `unsigned long long int*` cast): `readwrite`.
