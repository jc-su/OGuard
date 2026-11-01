# q22_core_cuda_q22_groupby_kernel

`kernel.cu` is the kernel `q22_groupby_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q22/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q22_groupby_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the plan's other kernels (`q22_avg_kernel`, `q22_filter_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q22_groupby_launcher`, called once by `_query_core` when `q22_filter_kernel` kept a row, launches `q22_groupby_kernel<<<(n_filtered + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `codes` (read, length `n_filtered`): `out_codes[:n_filtered]`, the country codes of the kept customers; `n_filtered` is the count `q22_filter_kernel` returned.
- `bals` (read, length `n_filtered`): `out_bals[:n_filtered]`, the balances of the kept customers.
- `n_filtered` (size): `out_count.item()` of `q22_filter_kernel`; the plan launches only when it is positive.
- `unique_codes` (read, length `n_unique`): `torch.unique(f_codes)`; `n_unique` is `unique_codes.numel()`.
- `numcust_out` (readwrite, length `n_unique`): `torch.zeros(n_unique, int32)`; the kernel adds to it with `atomicAdd`.
- `totacctbal_out` (readwrite, length `n_unique`): `torch.zeros(n_unique, float64)`; the kernel adds to it with `atomicAdd`.
- `n_unique` (size): `unique_codes.numel()`, at least 1 when `n_filtered` is.

Least values of the size symbols: `n_filtered` 1, `n_unique` 1. The plan launches only when the filter kept a row, so both are at least 1.
