# q17_core_cuda_q17_final_kernel

`kernel.cu` is the kernel `q17_final_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q17/round_10.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q17_final_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q17_final_launcher`, called once by `_query_core`, launches `q17_final_kernel<<<(n_sel + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `sel_idx` (index-in `avg_qty_ptr`, length `n_sel`): `idx_clamped[in_part]`, `torch.searchsorted` into the sorted part keys clamped to `n_parts - 1`, so every value lies in `[0, n_parts)`, the length of `avg_qty_ptr`; `n_sel` is `sel_idx.numel()`.
- `sel_qty` (read, length `n_sel`): `l_quantity[in_part]`, one per selected row.
- `sel_price` (read, length `n_sel`): `l_extendedprice[in_part]`, one per selected row.
- `avg_qty_ptr` (read, length `n_parts`): `avg_qty`, `n_parts = part_keys.numel()` elements, the filtered parts.
- `n_sel` (size): `sel_idx.numel()`.
- `out_ptr` (readwrite, length `1`): `torch::zeros({1})`; the kernel adds to it with `atomicAdd`.

Least values of the size symbols: `n_sel` 0, `n_parts` 1. The plan launches on whatever rows its membership test keeps, which can be none; `n_parts` is at least 1, since `sorted_keys[idx_clamped]` fails on an empty part list before the launch.
