# q06_cuda_q6_kernel

`kernel.cu` is the kernel `q6_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q06/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q6_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#define BLOCK_SIZE 1024`.
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q6_launcher` launches `q6_kernel<<<(n + 1023) / 1024, 1024>>>` once, with no dynamic shared memory.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `price_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `price.numel()`.
- `discount_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `price.numel()`.
- `quantity_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `price.numel()`.
- `shipdate_ptr` (read, length `n_elements`): `l_shipdate` in days since 1970-01-01, one per row.
- `date_lo` (scalar): from the query parameter; no range.
- `date_hi` (scalar): from the query parameter; no range.
- `disc_lo` (scalar): query parameter.
- `disc_hi` (scalar): query parameter.
- `qty_hi` (scalar): query parameter.
- `out_ptr` (readwrite, length `1`): `torch::zeros({1})` in the launcher; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `price.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
