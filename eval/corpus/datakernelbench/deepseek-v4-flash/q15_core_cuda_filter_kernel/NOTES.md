# q15_core_cuda_filter_kernel

`kernel.cu` is the kernel `filter_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q15/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `filter_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>`, `#define BLOCK_SIZE 256` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the plan's two earlier source strings (the first two assignments of `_CUDA_SOURCE`, one of which does not parse), which the third overwrites before `_load_cuda` compiles it.
- The kernel's body and parameters are the plan's.

## Launch

`filter_launcher`, called once by `_query_core`, launches `filter_kernel<<<(n + 255) / 256, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (256, 1, 1), 0 bytes of dynamic shared memory, and the grid maxima (2147483647, 1, 1): x up to the GPU's limit, since the 64-bit row index needs no bound, and one block in y and z, which the row index ignores.

## Lengths and ranges

- `l_suppkey` (read, length `n`): a `lineitem` column, one per row; `n` is `l_suppkey.numel()`.
- `l_extendedprice` (read, length `n`): a `lineitem` column, one per row.
- `l_discount` (read, length `n`): a `lineitem` column, one per row.
- `l_shipdate_days` (read, length `n`): `l_shipdate` in days, one per row.
- `date_lo` (scalar): from the query parameter; no range.
- `date_hi` (scalar): from the query parameter; no range.
- `n` (size): `l_suppkey.numel()`.
- `out_suppkey` (write, length `n`): `torch.empty(n, int64)` in `_query_core`.
- `out_revenue` (write, length `n`): `torch.empty(n, float64)` in `_query_core`.
- `out_count` (readwrite, length `1`): `torch.zeros(1, int32)`; the kernel takes output positions from it with `atomicAdd`.

Least values of the size symbols: `n` 0. The plan launches on the whole `lineitem` table, which can be empty.
