# q09_core_cuda_q9_amount_kernel

The kernel `q9_amount_kernel` of the CUDA plan for q09 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q09/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q9_amount_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 56: `q9_amount_kernel<<<blocks, BLOCK_SIZE>>>` in `q9_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The four columns are columns of the joined table, of `n = price.numel()` rows; `amount` is `torch::empty({n})`, written once per row: `write`.
