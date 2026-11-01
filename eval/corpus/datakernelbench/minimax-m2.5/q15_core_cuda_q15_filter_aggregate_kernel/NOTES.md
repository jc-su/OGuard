# q15_core_cuda_q15_filter_aggregate_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q15/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `q15_filter_aggregate_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <stdint.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q15_launcher` launches `q15_filter_aggregate_kernel<<<grid, BLOCK_SIZE>>>` with `grid = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `l_suppkey`: `n_rows` elements: one element per lineitem row, `n_rows = l_suppkey.size(0)`.
- `l_extendedprice`: `n_rows` elements: one element per lineitem row.
- `l_discount`: `n_rows` elements: one element per lineitem row.
- `l_shipdate_days`: `n_rows` elements: `l_shipdate` as days since 1970-01-01, one element per lineitem row.
- `sorted_suppkeys`: `n_suppliers` elements: `torch.sort(s_suppkey)`, one element per supplier row, `n_suppliers = sorted_suppkeys.size(0)`.
- `date_lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `date_hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `out_totals`: `torch.zeros(n_sup)`: `n_suppliers` elements, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n_rows`: `l_suppkey.size(0)`, the size symbol of the lineitem buffers.
- `n_suppliers`: `sorted_suppkeys.size(0)`, the size symbol of `sorted_suppkeys` and `out_totals`.
