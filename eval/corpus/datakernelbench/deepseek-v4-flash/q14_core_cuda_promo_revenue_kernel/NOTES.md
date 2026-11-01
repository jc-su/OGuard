# q14_core_cuda_promo_revenue_kernel

`kernel.cu` is the kernel `promo_revenue_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q14/round_07.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `promo_revenue_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#define BLOCK_SIZE 256` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`promo_launcher`, called once by `_query_core`, launches `promo_revenue_kernel<<<528, BLOCK_SIZE>>>` with no dynamic shared memory: a fixed grid of 528 blocks with a grid-stride loop. `launch.json` states the block (256, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `price.numel()`.
- `discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `shipdate_ptr` (read, length `n_elements`): `l_shipdate` in days, one per joined row.
- `promo_ptr` (read, length `n_elements`): `p_type_is_promo`, a bool per joined row.
- `date_lo` (scalar): from the query parameter; no range.
- `date_hi` (scalar): from the query parameter; no range.
- `total_out` (readwrite, length `1`): `torch::zeros({1})`; the kernel adds to it with `atomicAdd`.
- `promo_out` (readwrite, length `1`): `torch::zeros({1})`; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `price.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
