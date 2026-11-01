# q14_cuda_promo_revenue_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q14/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `promo_revenue_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Copied the macros that the kernel uses: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`promo_revenue_launcher` launches `promo_revenue_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `l_extendedprice`: `n_elements` elements: `l_extendedprice.numel()`, one element per joined row.
- `l_discount`: `n_elements` elements: one element per joined row.
- `l_shipdate_days`: `n_elements` elements: one element per joined row.
- `p_type_is_promo`: `n_elements` elements: one element per joined row.
- `shipdate_lo_days`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `shipdate_hi_days`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `total_revenue`: `torch::zeros({1})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `promo_revenue`: `torch::zeros({1})`, as `total_revenue`.
- `n_elements`: `n = l_extendedprice.numel()`, the size symbol of the row buffers.
