# q19_core_cuda_q19_kernel

`kernel.cu` is the kernel `q19_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q19/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q19_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q19_launcher`, called once by `_query_core`, launches `q19_kernel<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `l_quantity` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `l_quantity.numel()`.
- `l_extendedprice` (read, length `n_elements`): a column of the joined frame, one per row.
- `l_discount` (read, length `n_elements`): a column of the joined frame, one per row.
- `p_brand_id` (read, length `n_elements`): the brand code (12, 23, 34, or 0) of each joined row; the kernel only compares it.
- `p_container_grp` (read, length `n_elements`): the container group (1, 2, 3, or 0) of each joined row; the kernel only compares it.
- `p_size` (read, length `n_elements`): `p_size` of each joined row.
- `brand12_id` (scalar): the plan's constant 12; no range.
- `brand23_id` (scalar): the plan's constant 23; no range.
- `brand34_id` (scalar): the plan's constant 34; no range.
- `sm_grp` (scalar): the plan's constant 1; no range.
- `md_grp` (scalar): the plan's constant 2; no range.
- `lg_grp` (scalar): the plan's constant 3; no range.
- `out_ptr` (readwrite, length `1`): `torch::zeros({1})`; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `l_quantity.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
