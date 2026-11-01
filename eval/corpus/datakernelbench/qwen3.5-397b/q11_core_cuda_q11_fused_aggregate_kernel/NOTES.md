# q11_core_cuda_q11_fused_aggregate_kernel

The kernel `q11_fused_aggregate_kernel` of the CUDA plan for q11 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q11/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q11_fused_aggregate_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 103: `q11_fused_aggregate_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>` in `q11_aggregate_launcher`, when `n_unique_keys <= BLOCK_SIZE` (1024), with `blocks = (n_elements + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and `shared_mem_size = n_unique_keys * sizeof(double)` bytes of dynamic shared memory (`extern __shared__ double shared_vals[]`): `launch.json` states 8 bytes per `n_unique_keys` and the size maximum `n_unique_keys <= 1024` of this branch. With more distinct keys the plan launches `q11_simple_aggregate_kernel` instead.

## Lengths and ranges

- `supplycost`, `availqty`, and `inverse` have `n_elements = supplycost.numel()` rows (the launcher checks the other two); `inverse` is the inverse of `torch.unique(ps_partkey)`, in `[0, n_unique_keys)`: `index-in` `agg_ptr`.
- `agg` is `torch::zeros({n_unique_keys})`, an `atomicAdd` accumulator: `readwrite`.
