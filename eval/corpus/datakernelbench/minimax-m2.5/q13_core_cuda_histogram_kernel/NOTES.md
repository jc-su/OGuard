# q13_core_cuda_histogram_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q13/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `histogram_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cuda_runtime.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 256`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`histogram_add` launches `histogram_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 256`, and no dynamic shared memory, only when `n > 0` (the plan also calls it only when `o_custkey_filtered.numel() > 0`); `launch.json` states the block.

## Lengths and ranges

- `keys`: `n` elements: `o_custkey` of the orders whose comment does not match, one element per such order.
- `n`: `keys.numel()`, the size symbol of `keys`; the launcher returns before the launch when it is 0, so its least value is 1.
- `counts`: `torch.zeros(num_customers + 1)` with `num_customers = int(df_customer['c_custkey'].max())`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
