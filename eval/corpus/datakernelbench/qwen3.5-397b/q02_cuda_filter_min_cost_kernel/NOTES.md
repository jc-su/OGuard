# q02_cuda_filter_min_cost_kernel

The kernel `filter_min_cost_kernel` of the CUDA plan for q02 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q02/round_09.py`,
copied as `plan.py`).

## Extraction edits

- `filter_min_cost_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>`, `#include <limits>`, `#include <cmath>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 58: `filter_min_cost_kernel<<<blocks, BLOCK_SIZE>>>` in `filter_launcher`, with `blocks = (n_elements + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- `supplycost_ptr` and `min_supplycost_ptr` are two columns of the joined candidate table `cand`, of `n_elements = ps_supplycost.shape[0]` rows; the plan calls the launcher only when `len(cand) > 0`, so `n_elements` is at least 1.
- `keep_indices` is `torch::empty({n_elements})`: the kernel writes it at positions it takes from the counter.
- `keep_count` is `torch::zeros({1})`, a counter that the kernel increments with `atomicAdd`: `readwrite`.
