# q20_core_cuda_sum_by_group_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q20/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `sum_by_group_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cuda_runtime.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <stdio.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the other kernels of the source: `filter_and_collect_suppkeys_kernel`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`sum_by_group` launches `sum_by_group_kernel<<<grid_size, block_size>>>` with `grid_size = (N + block_size - 1) / block_size`, `block_size = BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `group_ids`: `N` elements, the filtered lineitem rows: `inverse[Nps:]` of `torch.unique(all_pairs, dim=0, return_inverse=True)`, so every value lies in [0, num_groups): an index into `output`.
- `quantities`: `N` elements: `l_quantity` of the same rows.
- `output`: `torch.zeros(num_groups)`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `N`: `N = group_ids.numel()`, the size symbol of the row buffers.
- `num_groups`: `unique_pairs.shape[0]`, the distinct (partkey, suppkey) pairs, the size symbol of `output`.
