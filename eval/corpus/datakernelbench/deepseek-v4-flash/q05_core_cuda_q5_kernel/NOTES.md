# q05_core_cuda_q5_kernel

`kernel.cu` is the kernel `q5_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q05/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q5_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q5_launcher`, called once by `_query_core`, launches `q5_kernel<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `price.numel()`.
- `discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `nation_id_ptr` (read, length `n_elements`): `final_df["nation_id"]`, the row number of `nations_unique` joined by `n_nationkey`; the kernel only compares it, so no index role.
- `out_ptr` (readwrite, length `num_nations`): `torch::zeros({num_nations})`; the kernel adds to it with `atomicAdd`.
- `num_nations` (size): `len(nations_unique)`, the length of `out_ptr`.
- `n_elements` (size): `price.numel()`.

Least values of the size symbols: `num_nations` 0, `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
