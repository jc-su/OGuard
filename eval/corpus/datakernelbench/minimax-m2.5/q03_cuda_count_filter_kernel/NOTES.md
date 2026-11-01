# q03_cuda_count_filter_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q03/round_05.py`, copied as `plan.py`.

## Extraction

- The kernel `count_filter_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cuda_runtime.h>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <cuda.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- Left out the other kernels of the source: `filter_project_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q3_filter_and_prepare` launches `count_filter_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024` threads per block, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `l_shipdate_days`: `n_elements` elements: `l_shipdate_days` has one element per row of the joined frame `col`, and `n = l_orderkey.numel()` is that row count.
- `o_orderdate_days`: `n_elements` elements: one element per row of `col`.
- `shipdate_lo_days`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `orderdate_hi_days`: a date bound, days since 1970-01-01 that the plan computes from its `orderdate_hi` parameter; any int32 value; not used in an address.
- `out_count`: 1 element: `count_tensor = torch::zeros({1}, kInt64)`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n_elements`: `n = l_orderkey.numel()`, the size symbol of the row buffers; least value 0.
