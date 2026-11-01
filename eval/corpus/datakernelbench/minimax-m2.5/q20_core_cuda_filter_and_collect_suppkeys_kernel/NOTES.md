# q20_core_cuda_filter_and_collect_suppkeys_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q20/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `filter_and_collect_suppkeys_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cuda_runtime.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <stdio.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the other kernels of the source: `sum_by_group_kernel`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`filter_and_collect_suppkeys` launches `filter_and_collect_suppkeys_kernel<<<grid_size, block_size>>>` with `grid_size = (N + block_size - 1) / block_size`, `block_size = BLOCK_SIZE = 1024`, and no dynamic shared memory, after `sum_by_group_kernel`; `launch.json` states the block.

## Lengths and ranges

- `group_ids`: `N` elements, the filtered partsupp rows: `inverse[:Nps]` of `torch.unique(all_pairs, dim=0, return_inverse=True)`, so every value lies in [0, num_groups): an index into `sum_qty`.
- `suppkeys`: `N` elements: `ps_suppkey` of the same rows.
- `availqty`: `N` elements: `ps_availqty` of the same rows.
- `sum_qty`: `num_groups` elements: the per-pair quantity sums that `sum_by_group_kernel` accumulated.
- `output`: `torch.full((N,), -1)`: `N` elements, which the kernel writes at every row.
- `N`: `N = group_ids.numel()`, the size symbol of the row buffers.
- `num_groups`: `unique_pairs.shape[0]`, the distinct (partkey, suppkey) pairs, the size symbol of `sum_qty`.
