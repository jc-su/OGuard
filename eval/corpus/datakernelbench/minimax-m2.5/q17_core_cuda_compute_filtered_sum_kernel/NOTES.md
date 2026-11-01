# q17_core_cuda_compute_filtered_sum_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q17/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `compute_filtered_sum_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Copied the macros that the kernel uses: `#define BLOCK_SIZE 1024`.
- Left out the other kernels of the source: `compute_part_sums_kernel`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`compute_filtered_sum` launches `compute_filtered_sum_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory, after `compute_part_sums_kernel`; `launch.json` states the block.

## Lengths and ranges

- `part_idx`: `n` elements, the lineitem rows whose part is in the filtered part set: `idx_clamped[in_part]`, the `searchsorted` position in `sorted_keys` clamped to `n_parts - 1`, so every value lies in [0, n_parts): an index into `threshold`.
- `quantity`: `n` elements: `l_quantity` of the same rows.
- `price`: `n` elements: `l_extendedprice` of the same rows.
- `threshold`: `0.2 * avg_qty`: `n_parts` elements, one per filtered part.
- `total`: `torch.zeros(1)`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n`: `n = part_idx.numel()`, the size symbol of the row buffers.
- `n_parts`: `sorted_keys.numel()`, the filtered parts, the size symbol of `threshold`.
