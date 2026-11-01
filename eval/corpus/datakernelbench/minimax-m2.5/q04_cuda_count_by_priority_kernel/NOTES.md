# q04_cuda_count_by_priority_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q04/round_06.py`, copied as `plan.py`.

## Extraction

- The kernel `count_by_priority_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <stdint.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`run_q4` launches `count_by_priority_kernel<<<256, BLOCK_SIZE>>>`: 256 blocks of `BLOCK_SIZE = 1024` threads, no dynamic shared memory, and a grid-stride loop over the orders.

## Lengths and ranges

- `o_orderkey`: `n_orders` elements: `o_orderkey.size(0)`, one element per order.
- `o_priority`: `n_orders` elements: one element per order.
- `o_orderdate`: `n_orders` elements: one element per order.
- `n_orders`: `n_orders = o_orderkey.size(0)`, the size symbol of the order buffers.
- `date_lo`: a date bound, days since 1970-01-01 that the plan computes from its `date_lo` parameter; any int32 value; not used in an address.
- `date_hi`: a date bound, days since 1970-01-01 that the plan computes from its `date_hi` parameter; any int32 value; not used in an address.
- `late_keys`: `n_late_keys` elements: `late_keys_unique`, the distinct keys of the late lineitems.
- `n_late_keys`: `n_late = late_keys_sorted.size(0)`, the size symbol of `late_keys`.
- `counts`: `torch::zeros({num_priorities})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `num_priorities`: `len(priority_categories)`, the size symbol of `counts`.
