# q17_cuda_q17_kernel

`kernel.cu` is the kernel `q17_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q17/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q17_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API) and `#include <cuda.h>` (the CUDA driver API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); and the plan's other kernels (`compute_part_stats_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q17_launcher` launches `q17_kernel<<<(n_lineitems + 1023) / 1024, 1024>>>` once, with no dynamic shared memory, after `compute_part_stats_kernel`.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `l_partkey_ptr` (read, length `n_lineitems`): `l_partkey`, one per `lineitem` row.
- `l_quantity_ptr` (read, length `n_lineitems`): a `lineitem` column, one per row.
- `l_extendedprice_ptr` (read, length `n_lineitems`): a `lineitem` column, one per row.
- `part_keys_ptr` (read, length `n_parts`): `p_partkey` of the filtered parts.
- `avg_qty_ptr` (read, length `n_parts`): `avg_qty = sum_qty / cnt_qty`, one per part.
- `threshold_ptr` (read, length `n_parts`): `threshold = 0.2 * avg_qty`, one per part; the kernel does not access it.
- `out_ptr` (readwrite, length `1`): `torch::zeros({1})`; thread 0 of each block adds to it with `atomicAdd`.
- `n_lineitems` (size): `l_partkey.numel()`.
- `n_parts` (size): `part_keys.numel()`.

Least values of the size symbols: `n_lineitems` 0, `n_parts` 0. The plan launches on whatever rows its joins and filters give, which can be none.
