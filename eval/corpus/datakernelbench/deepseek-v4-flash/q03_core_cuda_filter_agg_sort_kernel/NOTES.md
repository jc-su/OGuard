# q03_core_cuda_filter_agg_sort_kernel

`kernel.cu` is the kernel `filter_agg_sort_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q03/round_06.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `filter_agg_sort_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 256`, `#define WARP_SIZE 32` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); `#include <cuda.h>` (the CUDA driver API), which only the host code uses; the template `warp_reduce_sum`, which the kernel does not call.
- The kernel's body and parameters are the plan's.

## Launch

`query3_filter`, called once by `_query_core`, launches `filter_agg_sort_kernel<<<(n + 255) / 256, BLOCK_SIZE, smem_size>>>` with `smem_size = BLOCK_SIZE * (8 + 8 + 4 + 4)`: 256 threads and 6144 bytes of dynamic shared memory. `launch.json` states the block (256, 1, 1), 6144 bytes of dynamic shared memory, and the grid maxima (8388608, 1, 1): `2^31 / 256` blocks in x, past which the `int` row index `tid` wraps, and one block in y and z.

## Lengths and ranges

- `l_orderkey_ptr` (read, length `n`): `l_orderkey` of the joined frame, one per row; `n` is `l_orderkey.numel()`.
- `l_extendedprice_ptr` (read, length `n`): a column of the joined frame, one per row.
- `l_discount_ptr` (read, length `n`): a column of the joined frame, one per row.
- `l_shipdate_days_ptr` (read, length `n`): `l_shipdate` in days, one per joined row.
- `o_orderdate_days_ptr` (read, length `n`): `o_orderdate` in days, one per joined row.
- `o_shippriority_ptr` (read, length `n`): a column of the joined frame, one per row.
- `shipdate_lo_days` (scalar): from the query parameter; no range.
- `orderdate_hi_days` (scalar): from the query parameter; no range.
- `result_lk` (write, length `n`): `torch::empty({n})`.
- `result_rev` (write, length `n`): `torch::empty({n})`.
- `result_od` (write, length `n`): `torch::empty({n})`.
- `result_sp` (readwrite, length `n`): `torch::empty({n})`; the kernel takes output positions from `result_sp[n]` with `atomicAdd` and writes the kept rows' priorities into it.
- `n` (size): `l_orderkey.numel()`.
- `top_n` (scalar): the query parameter `top_n`; the kernel does not use it.

Least values of the size symbols: `n` 0. The plan launches on whatever rows its joins and filters give, which can be none.
