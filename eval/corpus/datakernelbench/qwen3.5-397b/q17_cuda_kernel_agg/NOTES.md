# q17_cuda_kernel_agg

The kernel `kernel_agg` of the CUDA plan for q17 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q17/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `kernel_agg` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- The device code that the kernel uses is copied unchanged: `__device__ __forceinline__ int64_t binary_search(`.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 132: `kernel_agg<<<blocks, BLOCK_SIZE>>>` in `q17_launcher`, with `blocks = (n_lineitems + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The lineitem columns have `n_lineitems = l_extendedprice.numel()` rows (the lineitem table's columns); `part_keys` is the sorted keys of the filtered parts, `n_parts = part_keys.numel()`. The launcher returns before the launch when either is 0, so both are at least 1.
- `sum_qty` and `count_qty` are `torch::zeros({n_parts})`, `atomicAdd` accumulators: `readwrite`.
