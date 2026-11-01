# q15_cuda_filter_aggregate_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q15/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `filter_aggregate_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cuda_runtime.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`filter_aggregate` launches `filter_aggregate_kernel<<<grid, BLOCK_SIZE>>>` with `grid = (n_lineitem + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `l_suppkey`: `n_lineitem` elements: `l_suppkey.size(0)`, one element per lineitem row.
- `l_price`: `n_lineitem` elements: one element per lineitem row.
- `l_discount`: `n_lineitem` elements: one element per lineitem row.
- `l_shipdate`: `n_lineitem` elements: one element per lineitem row.
- `date_lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `date_hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `sorted_suppkeys`: `n_supplier` elements: the sorted `s_suppkey` column.
- `out_revenue`: `torch.zeros(n_sup)`, one element per supplier, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n_lineitem`: `l_suppkey.size(0)`, the size symbol of the lineitem buffers.
- `n_supplier`: `sorted_suppkeys.size(0)` as an int, the size symbol of `sorted_suppkeys` and `out_revenue`.
