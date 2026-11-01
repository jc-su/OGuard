# q11_core_cuda_q11_total_sum_kernel

The kernel `q11_total_sum_kernel` of the CUDA plan for q11 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q11/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q11_total_sum_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 184: `q11_total_sum_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>` in `q11_total_sum_launcher`, with `blocks = (n_elements + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and `shared_mem_size = BLOCK_SIZE * sizeof(double)` = 8192 bytes of dynamic shared memory (`extern __shared__ double shared_sum[]`), which `launch.json` states.

## Lengths and ranges

- `supplycost` and `availqty` have `n_elements = supplycost.numel()` rows (the launcher checks the second).
- `total` is `torch::zeros({1})`, an `atomicAdd` accumulator: `readwrite`.
