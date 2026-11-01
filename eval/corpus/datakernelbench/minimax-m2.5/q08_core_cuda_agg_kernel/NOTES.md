# q08_core_cuda_agg_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q08/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `agg_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <stdint.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`agg_launcher` launches `agg_kernel<<<grid, BLOCK_SIZE>>>` with `grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `inv`: `n` elements, the rows of the joined frame: the inverse of `torch.unique(o_year, return_inverse=True)` cast to int32, so every value lies in [0, num_groups): an index into `sum_total`.
- `volume`: `n` elements: `l_extendedprice * (1 - l_discount)`, one element per joined row.
- `is_brazil`: `n` elements: the nation flag as uint8, one element per joined row.
- `sum_total`: `torch::zeros({num_groups})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `sum_brazil`: `num_groups` elements, as `sum_total`.
- `n`: `n = volume.numel()`, the size symbol of the row buffers.
- `num_groups`: `unique_years.shape[0]`, the distinct order years, the size symbol of the accumulators; the kernel does not read it.
