# q11_core_cuda_q11_simple_aggregate_kernel

The kernel `q11_simple_aggregate_kernel` of the CUDA plan for q11 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q11/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q11_simple_aggregate_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 113: `q11_simple_aggregate_kernel<<<blocks, BLOCK_SIZE>>>` in `q11_aggregate_launcher`, when `n_unique_keys > BLOCK_SIZE` (1024), with `blocks = (n_elements + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- `supplycost`, `availqty`, and `inverse` have `n_elements = supplycost.numel()` rows; `inverse` is the inverse of `torch.unique(ps_partkey)`, in `[0, n_unique_keys)`: `index-in` `agg_ptr`.
- `agg` is `torch::zeros({n_unique_keys})`, an `atomicAdd` accumulator: `readwrite`; this branch runs when `n_unique_keys > 1024`, so the symbol's least value is 1025.
