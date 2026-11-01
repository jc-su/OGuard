# q21_cuda_q21_fused_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q21/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `q21_fused_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <algorithm>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Copied the device function `find_order_idx` unchanged.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q21_launcher` launches `q21_fused_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n_l1 + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `l_orderkeys`: `n_l1` elements: `l1_orderkey`, one element per row of the frame `l1`.
- `l_late`: `n_l1` elements: one element per row of `l1`.
- `supp_ids`: `n_l1` elements, the `supp_id` of each row of `l1`. The plan assigns `supp_id = range(len(supp_n))` and joins it to `l1`, so each value lies in [0, num_suppliers): an index into `out_counts`.
- `lo_orderkeys`: `n_orders` elements: the sorted `l_orderkey` of the per-order aggregate `agg`.
- `lo_distinct_supp`: `n_orders` elements: one element per row of `agg`.
- `lo_late_distinct_supp`: `n_orders` elements: one element per row of `agg`.
- `n_l1`: `l_orderkeys.numel()`, the size symbol of the `l1` buffers.
- `n_orders`: `lo_orderkeys.numel()`, the size symbol of the `agg` buffers.
- `out_counts`: `torch::zeros({num_suppliers}, kInt64)` with `num_suppliers = len(supp_n)`, zeroed before the launch; the kernel adds to it with `atomicAdd` through an `unsigned long long` pointer.
