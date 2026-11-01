# q13_cuda_count_orders_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q13/round_04.py`, copied as `plan.py`.

## Extraction

- The kernel `count_orders_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- Left out the other kernels of the source: `gather_counts_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`count_orders_launcher` launches `count_orders_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `order_custkeys`: `n_orders` elements: `o_custkey` of the orders whose comment does not match, one element per such order.
- `histogram`: `torch.zeros(num_customers + 1)` with `num_customers = int(df_customer['c_custkey'].max())`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n_orders`: `o_custkey_filtered.size(0)`, the size symbol of `order_custkeys`.
