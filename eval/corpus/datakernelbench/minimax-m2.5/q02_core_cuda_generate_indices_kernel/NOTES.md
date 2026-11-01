# q02_core_cuda_generate_indices_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q02/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `generate_indices_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`generate_indices_launcher` launches `generate_indices_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE`, `const int BLOCK_SIZE = 256` (a local of the launcher), and no dynamic shared memory, only when `k > 0`; `launch.json` states the block.

## Lengths and ranges

- `out_ptr`: `torch::empty({k})`: `k` elements, which the kernel writes.
- `n_elements`: `n = input_tensor.numel()`, the rows of the sorted result frame `final`; the kernel compares `idx` with it and does not use it in an address; any int64 value.
- `k`: `min(limit, n)`, the size symbol of `out_ptr`; the launcher launches only when it is positive, so its least value is 1.
