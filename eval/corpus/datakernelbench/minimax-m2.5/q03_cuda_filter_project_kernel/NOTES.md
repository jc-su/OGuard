# q03_cuda_filter_project_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q03/round_05.py`, copied as `plan.py`.

## Extraction

- The kernel `filter_project_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cuda_runtime.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <cuda.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- Left out the other kernels of the source: `count_filter_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q3_filter_and_prepare` launches `filter_project_kernel<<<blocks, BLOCK_SIZE>>>` after `count_filter_kernel`, with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory, only when `filtered_count > 0`.

## Lengths and ranges

- `l_orderkey`: `n_elements` elements: one element per row of the joined frame `col`.
- `l_extendedprice`: `n_elements` elements: one element per row of `col`.
- `l_discount`: `n_elements` elements: one element per row of `col`.
- `l_shipdate_days`: `n_elements` elements: one element per row of `col`.
- `o_orderdate_days`: `n_elements` elements: one element per row of `col`.
- `o_shippriority`: `n_elements` elements: one element per row of `col`.
- `shipdate_lo_days`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `orderdate_hi_days`: a date bound, days since 1970-01-01 that the plan computes from its `orderdate_hi` parameter; any int32 value; not used in an address.
- `out_orderkey`: `torch::empty({filtered_count})`, where `filtered_count` is the count that `count_filter_kernel` returned; the plan launches only when it is at least 1. The signature cannot relate it to the matching rows.
- `out_orderdate`: `filtered_count` elements, as `out_orderkey`.
- `out_shippriority`: `filtered_count` elements, as `out_orderkey`.
- `out_revenue`: `filtered_count` elements, as `out_orderkey`.
- `out_write_idx`: 1 element: `write_idx = torch::zeros({1}, kInt64)`, zeroed before the launch; the kernel takes positions from it with `atomicAdd`.
- `n_elements`: `n = l_orderkey.numel()`; at least 1, since `filtered_count` is at least 1.
