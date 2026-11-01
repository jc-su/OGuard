# q13_cuda_gather_counts_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q13/round_04.py`, copied as `plan.py`.

## Extraction

- The kernel `gather_counts_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- Left out the other kernels of the source: `count_orders_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`gather_counts_launcher` launches `gather_counts_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `custkeys`: `n_customers` elements, the whole `c_custkey` column. The plan sizes `histogram` as the greatest of these keys plus 1, and TPC-H customer keys are positive, so each value lies in [0, num_customers]: an index into `histogram`.
- `histogram`: `torch.zeros(num_customers + 1)` with `num_customers = int(df_customer['c_custkey'].max())`, filled by `count_orders_kernel`.
- `counts`: `n_customers` elements: `torch.empty_like(c_custkey)`.
- `n_customers`: `c_custkey.size(0)`, the size symbol of `custkeys` and `counts`.
