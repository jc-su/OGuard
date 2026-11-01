# q17_cuda_compute_part_stats_kernel

`kernel.cu` is the kernel `compute_part_stats_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q17/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `compute_part_stats_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API) and `#include <cuda.h>` (the CUDA driver API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); and the plan's other kernels (`q17_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q17_launcher` launches `compute_part_stats_kernel<<<(n_lineitems + 1023) / 1024, 1024>>>` once, with no dynamic shared memory, before `q17_kernel`.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `l_partkey_ptr` (read, length `n_lineitems`): `l_partkey`, one per `lineitem` row; `n_lineitems` is `l_partkey.numel()`.
- `l_quantity_ptr` (read, length `n_lineitems`): `l_quantity`, one per row.
- `part_keys_ptr` (read, length `n_parts`): `p_partkey` of the filtered parts; `n_parts` is `part_keys.numel()`.
- `sum_qty_ptr` (readwrite, length `n_parts`): `torch::zeros({n_parts})` in the launcher; the kernel adds to it with `atomicAdd`.
- `cnt_qty_ptr` (readwrite, length `n_parts`): `torch::zeros({n_parts})` in the launcher; the kernel adds to it with `atomicAdd`.
- `n_lineitems` (size): `l_partkey.numel()`.
- `n_parts` (size): `part_keys.numel()`.

Least values of the size symbols: `n_lineitems` 0, `n_parts` 0. The plan launches on whatever rows its joins and filters give, which can be none.
