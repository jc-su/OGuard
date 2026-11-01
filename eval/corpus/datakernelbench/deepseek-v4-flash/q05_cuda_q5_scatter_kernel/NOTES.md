# q05_cuda_q5_scatter_kernel

`kernel.cu` is the kernel `q5_scatter_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q05/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q5_scatter_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q5_launcher` launches `q5_scatter_kernel<<<(n + 1023) / 1024, 1024>>>` once, with no dynamic shared memory; `run_query` calls it once.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `nation_id_ptr` (index-in `out_ptr`, length `n_elements`): `final_df["nation_id"]`: `nation_id` is the row number `range(len(nations_unique))` of `nations_unique`, joined to each row by `n_nationkey`, so every value lies in `[0, num_nations)`, the length of `out_ptr`.
- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `price.numel()`.
- `discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `price.numel()`.
- `out_ptr` (readwrite, length `num_nations`): `torch.zeros(num_nations)`, `num_nations = len(nations_unique)`; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `price.numel()`.

Least values of the size symbols: `n_elements` 0, `num_nations` 0. The plan launches on whatever rows its joins and filters give, which can be none.
