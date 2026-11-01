# q19_cuda_q19_kernel

`kernel.cu` is the kernel `q19_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q19/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q19_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#define BLOCK_SIZE 1024`.
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q19_launcher` launches `q19_kernel<<<(n + 1023) / 1024, 1024>>>` once, with no dynamic shared memory.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `quantity_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `quantity.numel()`.
- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `quantity.numel()`.
- `discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `quantity.numel()`.
- `brand_id_ptr` (read, length `n_elements`): a code or `p_size` of each joined row.
- `container_grp_ptr` (read, length `n_elements`): a code or `p_size` of each joined row.
- `size_ptr` (read, length `n_elements`): a code or `p_size` of each joined row.
- `brand12_id` (scalar): the constant 12 of `run_query`.
- `brand23_id` (scalar): the constant 23.
- `brand34_id` (scalar): the constant 34.
- `sm_grp` (scalar): the constant 1.
- `md_grp` (scalar): the constant 2.
- `lg_grp` (scalar): the constant 3.
- `out_ptr` (readwrite, length `1`): `torch::zeros({1})`; thread 0 of each block adds to it with `atomicAdd`.
- `n_elements` (size): `quantity.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
