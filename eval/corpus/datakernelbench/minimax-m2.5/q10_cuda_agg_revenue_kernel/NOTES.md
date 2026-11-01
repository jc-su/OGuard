# q10_cuda_agg_revenue_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q10/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `agg_revenue_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <stdint.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`agg_revenue_launcher` launches `agg_revenue_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `keys`: `n` elements, the `c_custkey` of each joined row. The plan sizes `out` from the greatest of these keys (`max_key + 1` elements), and TPC-H customer keys are positive, so each value lies in [0, max_key]: an index into `out`.
- `price`: `n` elements: one element per joined row.
- `discount`: `n` elements: one element per joined row.
- `out`: `torch.zeros(max_key + 1)` with `max_key = int(c_custkey.max())`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n`: `keys.numel()`; `_query_core` returns before the launch when it is 0, so its least value is 1.
