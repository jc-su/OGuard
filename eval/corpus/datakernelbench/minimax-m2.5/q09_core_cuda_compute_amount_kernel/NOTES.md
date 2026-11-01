# q09_core_cuda_compute_amount_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q09/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `compute_amount_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`compute_amount_launcher` launches `compute_amount_kernel<<<grid_size, block_size>>>` with `grid_size = (n + block_size - 1) / block_size`, `block_size = BLOCK_SIZE = 1024`, and no dynamic shared memory, only when `n > 0`; `launch.json` states the block.

## Lengths and ranges

- `price`: `n` elements: `l_extendedprice` of the joined frame, `n = price.numel()` rows.
- `discount`: `n` elements: one element per joined row.
- `supplycost`: `n` elements: one element per joined row.
- `quantity`: `n` elements: one element per joined row.
- `out`: `torch::empty({n})`: `n` elements, which the kernel writes.
- `n`: `n = price.numel()`, the size symbol of every buffer; the launcher returns before the launch when it is 0, so its least value is 1.
