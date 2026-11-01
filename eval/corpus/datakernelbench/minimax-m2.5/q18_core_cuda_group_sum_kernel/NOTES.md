# q18_core_cuda_group_sum_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q18/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `group_sum_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cuda_runtime.h>`, `#include <stdint.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`group_sum_launcher` launches `group_sum_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `qty`: `n` elements: `l_quantity` in the order of the sorted order keys, `n = qty.numel()` rows.
- `group_idx`: `n` elements: `cumsum(boundaries) - 1`, the running count of group starts, whose last value is `num_groups - 1` with `num_groups = boundaries.sum()`, so every value lies in [0, num_groups): an index into `group_sum`.
- `group_sum`: `torch::zeros({num_groups})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n`: `n = qty.numel()`, the size symbol of the row buffers.
- `num_groups`: the distinct order keys, the size symbol of `group_sum`; the launcher takes it as an argument that the kernel does not read.
