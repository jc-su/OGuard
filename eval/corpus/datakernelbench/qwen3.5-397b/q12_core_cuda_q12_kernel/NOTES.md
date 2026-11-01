# q12_core_cuda_q12_kernel

The kernel `q12_kernel` of the CUDA plan for q12 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q12/round_04.py`,
copied as `plan.py`).

## Extraction edits

- `q12_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers; `<cstdint>` is included in its place for `int32_t` and `int64_t`), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 102: `q12_kernel<<<blocks, BLOCK_SIZE>>>` in `q12_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory (`s_counts` is static).

## Lengths and ranges

- The five columns are columns of the joined lineitem and orders table, of `n = shipmode.numel()` rows; `shipmode` and `priority` are 0/1 flags, which the kernel checks itself (`mode` in `[0, 2)`, `high` as a boolean).
- `out` is `torch::zeros({4})` of int64, passed as `unsigned long long*`, an `atomicAdd` accumulator: `readwrite`.
