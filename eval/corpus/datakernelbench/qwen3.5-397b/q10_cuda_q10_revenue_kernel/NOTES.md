# q10_cuda_q10_revenue_kernel

The kernel `q10_revenue_kernel` of the CUDA plan for q10 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q10/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q10_revenue_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 61: `q10_revenue_kernel<<<blocks, BLOCK_SIZE>>>` in `q10_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The columns have `n = custkey.numel()` rows; `inverse` comes from `torch.unique(c_custkey, return_inverse=True)` and has as many rows.
- `revenue` is `torch.zeros(unique_keys.shape[0])`: `n_unique` elements, an `atomicAdd` accumulator (`readwrite`); the inverse indices of `torch.unique` lie in `[0, n_unique)`, so `inverse_ptr` is `index-in` `revenue_ptr`.
